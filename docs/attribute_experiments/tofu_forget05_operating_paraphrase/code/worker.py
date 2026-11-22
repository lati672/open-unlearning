"""Bounded reproduction and question-only paraphrase evaluation using frozen Table4 code."""
import os,sys,json,hashlib,math,time,copy
from pathlib import Path
from datetime import datetime,timezone
REPO=Path('/workspace/open-unlearning')
R=REPO/'saves/experiments/tofu_forget05_attribute_operating_paraphrase_v1'
SOURCE=REPO/'saves/experiments/tofu_forget05_adaptive_density_phase_a/source_v4'
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'scripts'),str(REPO/'scripts')]
import torch
import numpy as np
from omegaconf import OmegaConf
from data import get_data
from data.collators import DataCollatorForSupervisedDataset,FrozenMaskCollator
from data.qa import QADataset
from data.utils import preprocess_chat_instance
from model import get_model
from trainer import load_trainer
from trainer.masking import eligible_positions
from trainer.utils import seed_everything
from train_mask_experiment import StepCheckpointEvaluation
from run_low_confidence import instrument,replay_metrics
from evaluate_tofu_checkpoint import validate_outputs,get_evaluators
from datasets import load_dataset,Dataset
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def frozen():
 for n,h in read(R/'paraphrases/FROZEN.json')['files_sha256'].items():assert sha(R/'paraphrases'/n)==h,n
 for n,h in read(R/'manifests/source_hashes.json').items():assert sha(SOURCE/n)==h,n
def prior(agg,method,seed):
 exp='tofu_forget05_attribute_only'+('_token_mean' if agg=='token_mean' else '')
 return REPO/'saves/experiments'/exp/'jobs'/method/'runs'/f'seed{seed}'
def runpath(agg,method,seed):return R/'runs'/agg/method/f'seed{seed}'
class Collator(FrozenMaskCollator):
 def __init__(self,tokenizer,method):
  f=R/'manifests/frozen_inputs'
  super().__init__(DataCollatorForSupervisedDataset(tokenizer,index='index'),f/'AttributeOnly.jsonl',f/'RandomAttribute.jsonl' if method=='RandomAttribute' else None)
  self.order=hashlib.sha256();self.consumed=[]
 def __call__(self,items):
  b=super().__call__(items);f=b['forget'];f['entity_mask']=f['reference_mask']
  a=f['reference_mask'];c=f.get('frozen_random_mask',a);eligible=eligible_positions(f)
  assert a.shape==c.shape==eligible.shape and torch.equal(a.sum(-1),c.sum(-1)) and not ((a|c)&~eligible).any()
  return b
 def record(self,b):
  ids=[b[s]['index'].tolist() for s in ['forget','retain']]
  self.order.update(json.dumps(ids,separators=(',',':')).encode());self.consumed.append(ids)
def train(agg,method,seed):
 frozen();out=runpath(agg,method,seed);assert not out.exists() or not any(out.iterdir()),'Never overwrite/automatically resume an incomplete run'
 assert int(os.environ['WORLD_SIZE'])==2
 old=prior(agg,method,seed);snap=read(old/'resolved_config.json');oldargs=read(old/'training_args.json')
 cfg=OmegaConf.create(snap['trainer']);cfg.args.output_dir=str(out);cfg.args.logging_dir=str(out/'logs')
 seed_everything(seed)
 model,tok=get_model(OmegaConf.create(snap['model']));model.config.use_cache=False
 assert model.config._commit_hash=='88e31200b97e4c0c04ae0d2f0b591f427046d192'
 data=get_data(OmegaConf.create(snap['data']),mode='unlearn',tokenizer=tok,template_args=snap['model']['template_args'])['train']
 for split,n in [('forget',200),('retain',3800)]:
  ds=getattr(data,split);assert len(ds)==n
  assert all('324592d84ae4f482ac7249b9285c2ecdb53e3a68' in Path(x['filename']).parts for x in ds.data.cache_files)
 collator=Collator(tok,method)
 # Audits every frozen input before any optimizer update.
 counts=[];elig=0
 for i in range(200):
  item=data.forget[i];record=collator.records[i];rnd=json.loads((R/'manifests/frozen_inputs/RandomAttribute.jsonl').read_text().splitlines()[i])
  assert item['input_ids'].tolist()==record['input_ids']==rnd['input_ids']
  e=eligible_positions({k:v.unsqueeze(0) for k,v in item.items() if k in ['input_ids','attention_mask','labels']})[0]
  a=torch.tensor(record['mask']).bool();b=torch.tensor(rnd['mask']).bool()
  assert len(a)==len(b)==len(e) and a.sum()==b.sum() and not ((a|b)&~e).any()
  counts.append(int(a.sum()));elig+=int(e.sum())
 assert sum(counts)==1432 and counts.count(0)==15
 trainer,args=load_trainer(cfg,model,train_dataset=data,tokenizer=tok,data_collator=collator,template_args=snap['model']['template_args'])
 actual=args.to_dict();ignored={'output_dir','logging_dir','run_name','local_rank'}
 diff={k:[oldargs.get(k),actual.get(k)] for k in oldargs.keys()|actual.keys() if k not in ignored and oldargs.get(k)!=actual.get(k)}
 assert not diff,diff
 assert args.max_steps==120 and args.warmup_steps==6 and trainer.normalization==agg and trainer.accelerator.num_processes==2
 assert all(torch.equal(a,b) for a,b in zip(model.parameters(),trainer.ref_model.parameters(),strict=True))
 # Original code uses torch.no_grad for reference forwards; it does not clear
 # every requires_grad flag. Preserve its behavior and verify actual immutability.
 ref_versions=[p._version for p in trainer.ref_model.parameters()]
 assert {id(p) for p in trainer.ref_model.parameters()}.isdisjoint(id(p) for p in model.parameters())
 schedule=dict(max_steps=120,save_steps=60,checkpoints={str(x):x*.16 for x in [30,60,90,120]})
 snap['trainer']=OmegaConf.to_container(cfg,resolve=True);snap['paths']['output_dir']=str(out)
 snap['frozen_protocol']['checkpoints']=schedule['checkpoints']
 snap['execution']['trajectory_kind']='fresh reproduction; not continuation'
 snap['mask_file']=str(R/'manifests/frozen_inputs'/f'{method}.jsonl')
 snap['new_run_manifest_sha256']=sha(R/'manifests/run_manifest.json')
 snap['paraphrase_frozen_sha256']=sha(R/'paraphrases/FROZEN.json')
 original_loss=trainer.compute_loss
 def tracked(model,inputs,return_outputs=False):
  collator.record(inputs)
  return original_loss(model,inputs,return_outputs=return_outputs)
 trainer.compute_loss=tracked;checks=instrument(trainer)
 callback=StepCheckpointEvaluation(trainer,schedule,None,out,resolved_config=snap,defer_evaluation=True)
 trainer.add_callback(callback)
 if trainer.is_world_process_zero():
  out.mkdir(parents=True,exist_ok=True);save(out/'resolved_config.json',snap);save(out/'training_args.json',actual)
  save(out/'mask_input_audit.json',dict(samples=200,selected=sum(counts),eligible=elig,empty=counts.count(0),per_example_count=counts,input_ids_original_match=True,eligibility_passed=True))
 trainer.accelerator.wait_for_everyone();seed_everything(seed);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
 result=trainer.train();elapsed=time.monotonic()-t;trainer.save_state()
 assert trainer.state.global_step==120 and math.isfinite(result.training_loss) and len(checks['loss'])==480 and len(checks['gradient_l2_preclip'])==120
 assert [p._version for p in trainer.ref_model.parameters()]==ref_versions and all(p.grad is None for p in trainer.ref_model.parameters())
 assert {id(p) for p in trainer.ref_model.parameters()}.isdisjoint(id(p) for g in trainer.optimizer.param_groups for p in g['params'])
 original_order=read(old/f'finite_checks_rank{args.process_index}.json')['sample_order_sha256']
 assert collator.order.hexdigest()==original_order,('Data order differs',collator.order.hexdigest(),original_order)
 for i,(_,m) in trainer.mask_selector.cache.items():
  record=(collator.random_records if method=='RandomAttribute' else collator.records)[i]
  assert m.int().tolist()==record['mask'],i
 save(out/f'finite_checks_rank{args.process_index}.json',dict(**checks,sample_order_sha256=collator.order.hexdigest(),same_as_original_order=True,consumed_batches=collator.consumed,forget_exposures=1920,retain_exposures=1920,reference_initial_bitwise_equal=True,reference_frozen=True,elapsed_seconds=elapsed,gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
 trainer.accelerator.wait_for_everyone()
 if trainer.is_world_process_zero():
  assert set(callback.saved)=={'30','60','90','120'}
  for st in [30,60,90,120]:
   cp=out/f'checkpoint-{st}'
   save(cp/'weight_hashes.json',{p.name:sha(p) for p in cp.glob('*.safetensors')})
  save(out/'training_completed.json',dict(global_step=120,training_runtime_seconds=elapsed,train_loss=result.training_loss,seed=seed,method=method,aggregation=agg,kind='fresh reproduction',same_original_data_order=True,weights_preserved_pending_all_evaluations=True))
def official(agg,method,seed,step):
 frozen();cp=runpath(agg,method,seed)/f'checkpoint-{step}';snap=read(cp/'resolved_config.json')
 state=read(cp/'trainer_state.json');meta=read(cp/'checkpoint_metadata.json');assert state['global_step']==meta['global_step']==step and meta['seed']==seed and meta['mask_sha256']==snap['mask_sha256']
 for n,h in read(cp/'weight_hashes.json').items():assert sha(cp/n)==h
 out=R/'evaluations/reproduction'/agg/method/f'seed{seed}/step{step}/official';out.mkdir(parents=True,exist_ok=True)
 prov=read(prior(agg,method,seed)/'checkpoint-120/evals/official/evaluation_provenance.json')
 cfg=copy.deepcopy(prov['resolved_eval_config']);cfg['output_dir']=str(out)
 ref=REPO/'saves/experiments/tofu_forget05_multiseed_v4_official/frozen_inputs/TOFU_EVAL.json'
 protocol=read(R/'protocol.json')['original_protocol']
 assert sha(ref)==protocol['inputs'][protocol['retain_only_model']['evaluation_logs']]
 signature={k:v for k,v in cfg.items() if k not in ['output_dir','retain_logs_path']}
 assert signature==read(R/'manifests'/agg/'v4_evaluation_signature.json')
 save(out/'evaluation_provenance.json',dict(checkpoint=meta,weight_sha256=read(cp/'weight_hashes.json'),resolved_eval_config=cfg,source_code_hashes_sha256=sha(R/'manifests/source_hashes.json'),kind='new measurement on fresh reproduction',attention_backend=snap['model']['model_args']['attn_implementation'],precision='BF16',batch_size=32))
 mc=OmegaConf.create(snap['model']);mc.model_args.pretrained_model_name_or_path=str(cp);mc.model_args.pop('revision',None);mc.model_args.local_files_only=True;mc.model_args.device_map={'':'cuda:0'}
 model,tok=get_model(mc);model.eval();t=time.monotonic();torch.cuda.reset_peak_memory_stats()
 if not (out/'TOFU_EVAL.json').exists():get_evaluators(OmegaConf.create({'tofu':cfg}))['tofu'].evaluate(model=model,tokenizer=tok,template_args=snap['model']['template_args'])
 logs=read(out/'TOFU_EVAL.json');summary=read(out/'TOFU_SUMMARY.json')
 metrics,counts=validate_outputs(cfg,protocol['official_evaluation'],logs,summary,ref);recomputed=replay_metrics(cfg,logs,read(ref))
 assert all(math.isclose(metrics[k],recomputed[k],rel_tol=1e-10,abs_tol=1e-14) for k in metrics)
 save(out/'official_metrics.json',metrics)
 save(out/'validated.json',dict(aggregation=agg,method=method,seed=seed,step=step,metrics=metrics,recomputed_metrics=recomputed,counts=counts,elapsed_seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_finite=True,checkpoint_sha256=read(cp/'weight_hashes.json')))
 if step==120:paraphrase(agg,method,seed,model,tok,snap,out.parent/'questions')
 print('OFFICIAL COMPLETE',agg,method,seed,step,metrics,flush=True)
def paraphrase(agg,method,seed,model=None,tok=None,snap=None,out=None):
 from evals.metrics.utils import evaluate_probability,eval_text_similarity
 from torch.utils.data import DataLoader
 frozen()
 if snap is None:
  assert agg=='target';snap=read(prior('summed','AttributeOnly',0)/'resolved_config.json')
  mc=OmegaConf.create(snap['model']);mc.model_args.device_map={'':'cuda:0'};model,tok=get_model(mc);model.eval()
  out=R/'evaluations/original_target/questions'
 out.mkdir(parents=True,exist_ok=True)
 rev=read(R/'protocol.json')['original_protocol']['dataset']['revision']
 ds=load_dataset('locuslab/TOFU','forget05_perturbed',revision=rev,split='train')
 raw=load_dataset('locuslab/TOFU','forget05',revision=rev,split='train')
 assert len(ds)==len(raw)==200 and all(ds[i]['question']==raw[i]['question'] and ds[i]['answer']==raw[i]['answer'] for i in range(200))
 frozen_rows=[json.loads(x) for x in (R/'paraphrases/questions.jsonl').read_text().splitlines()]
 template=snap['model']['template_args'];summary={};t=time.monotonic()
 for style in ['original','ordinary','substantial']:
  outfile=out/f'{style}.json'
  if outfile.exists():
   summary[style]=read(outfile)['summary'];continue
  recs=[dict(question=x['question'],answer=x['answer']) for x in ds]
  if style!='original':
   selected=[x for x in frozen_rows if x['style']==style];assert [x['original_id'] for x in selected]==list(range(200))
   for i,x in enumerate(selected):assert x['original_question']==recs[i]['question'];recs[i]['question']=x['question']
  qa=QADataset(dict(path='locuslab/TOFU',name='forget05_perturbed',revision=rev,split='train'),template,tok,question_key='question',answer_key='answer',max_length=512,verify_revision=True)
  qa.data=Dataset.from_list([{**x,'index':i} for i,x in enumerate(recs)])
  gen=copy.copy(qa);gen.predict_with_generate=True
  audits=[]
  for i in range(200):
   a=qa[i];b=gen[i];q=recs[i]['question']
   empty=preprocess_chat_instance(tok,template,[q],[''],512,True)
   assert torch.equal(b['input_ids'],empty['input_ids'])
   n=len(b['input_ids']);assert torch.equal(a['input_ids'][:n],b['input_ids']) and a['labels'][:n].eq(-100).all() and a['labels'][n:].ne(-100).all()
   audits.append(dict(original_id=i,prompt_length=n,answer_label_count=int(a['labels'].ne(-100).sum()),generation_input_ids_sha256=hashlib.sha256(b['input_ids'].numpy().tobytes()).hexdigest(),generation_independent_of_answer=True,prompt_answer_prefix_exact=True))
  rows={}
  coll=DataCollatorForSupervisedDataset(tok,index='index')
  with torch.inference_mode():
   for batch in DataLoader(qa,batch_size=32,collate_fn=coll):
    ids=batch.pop('index').tolist();values=evaluate_probability(model,batch,record_details=True)
    for i,v in zip(ids,values):rows[i]=dict(original_id=i,question=recs[i]['question'],target_answer=recs[i]['answer'],**v)
  coll=DataCollatorForSupervisedDataset(tok,padding_side='left',index='index')
  generation=OmegaConf.create(dict(do_sample=False,top_p=None,temperature=None,max_new_tokens=200,use_cache=True))
  with torch.inference_mode():
   for batch in DataLoader(gen,batch_size=32,collate_fn=coll):
    ids=batch.pop('index').tolist();values=eval_text_similarity(model,tok,batch,generation)
    for i,v in zip(ids,values):rows[i].update(v)
  vals=list(rows.values());assert len(vals)==200 and all(math.isfinite(x[k]) for x in vals for k in ['prob','avg_loss','rougeL_recall'])
  for x in vals:assert len(x['token_logprobs'])==x['token_count'] and math.isclose(math.exp(sum(x['token_logprobs'])/x['token_count']),x['prob'],rel_tol=2e-5,abs_tol=1e-7)
  summary[style]=dict(F_Prob=float(np.mean([x['prob'] for x in vals])),F_ROUGE=float(np.mean([x['rougeL_recall'] for x in vals])),n_original_ids=200)
  save(outfile,dict(aggregation=agg,method=method,seed=None if agg=='target' else seed,style=style,summary=summary[style],rows=vals))
  save(out/f'{style}_input_audit.json',dict(passed=True,per_item=audits,no_answer_in_generation_input=True,official_probability_and_generation_function=True))
  print('PARAPHRASE',agg,method,seed,style,summary[style],flush=True)
 if agg!='target':
  official_logs=read(out.parent/'official/TOFU_EVAL.json')
  base=read(out/'original.json')
  assert math.isclose(base['summary']['F_Prob'],official_logs['forget_Q_A_Prob']['agg_value'],rel_tol=1e-12)
  assert math.isclose(base['summary']['F_ROUGE'],official_logs['forget_Q_A_ROUGE']['agg_value'],rel_tol=1e-12)
 save(out/'completed.json',dict(summary=summary,elapsed_seconds=time.monotonic()-t,all200_included=True,ambiguity_sensitivity_ids_excluded=[117,131],batch_size=32,attention_backend='flash_attention_2',precision='BF16',greedy_max_new_tokens=200,frozen_questions_sha256=sha(R/'paraphrases/FROZEN.json')))
def main():
 action=sys.argv[1]
 if action=='target':paraphrase('target','original_target',0)
 else:
  agg,method,seed=sys.argv[2:5];seed=int(seed)
  assert agg in ['summed','token_mean'] and method in ['AttributeOnly','RandomAttribute'] and seed in range(3)
  if action=='train':train(agg,method,seed)
  elif action=='evaluate':official(agg,method,seed,int(sys.argv[5]))
  else:raise ValueError(action)
if __name__=='__main__':main()
