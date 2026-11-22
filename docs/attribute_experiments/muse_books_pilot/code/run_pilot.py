"""Hard-bounded MUSE Books attribute pilot: only seed0, two arms, <=4 updates."""
import argparse,copy,datetime,hashlib,json,math,os,re,sys,time
from pathlib import Path
P=Path('/workspace/open-unlearning');OLD=P/'saves/experiments/muse_books_rmu_multiseed_v1'
SRC=OLD/'source';R=P/'saves/experiments/muse_books_attribute_pilot_v1'
sys.path.insert(0,str(SRC/'src'))
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from model import get_model,get_tokenizer
from data.collators import DataCollatorForSupervisedDataset
from trainer.masking import eligible_positions
from trainer.unlearn.rmu import RMU
from transformers import set_seed

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def locked():
 frozen=read(R/'FROZEN.json')
 assert all(sha(R/n)==v for n,v in frozen['files'].items()),'Frozen annotations, inputs or masks changed'
 plan=read(R/'plan.json');assert plan['max_optimizer_updates_per_arm']==4 and plan['training_seeds']==[0]
 assert all(sha(SRC/n)==v for n,v in plan['source_sha256'].items())
 assert read(R/'inputs/target_download_verified.json')['all_six_shards_sha256_verified']
 return plan
def inputs():
 samples={role:{x['doc_id']:x for x in map(json.loads,(R/'inputs'/f'{role}_samples.jsonl').read_text().splitlines())} for role in ['forget','retain']}
 masks={arm:{x['doc_id']:x for x in map(json.loads,(R/f'masks_{arm}.jsonl').read_text().splitlines())} for arm in ['attribute','random_attribute']}
 return samples,masks
def load():
 set_seed(0);c=OmegaConf.create(copy.deepcopy(read(R/'plan.json')['model']))
 c.model_args.device_map={'':'cuda:0'}
 model,tokenizer=get_model(c);model.config.use_cache=False
 assert model.config.hidden_size==4096 and next(model.parameters()).dtype==torch.bfloat16
 model.eval()
 return model,tokenizer
def batch(records,ids,collator):
 items=[{k:torch.tensor(records[i][k]) for k in ['input_ids','labels','attention_mask']}|{'index':i} for i in ids]
 return {k:v.to('cuda:0') for k,v in collator(items).items()}
def hidden(model,b,grad=False):
 cache=[]
 def hook(module,args,output):cache.append(output[0] if isinstance(output,tuple) else output)
 handle=model.model.layers[7].register_forward_hook(hook)
 try:
  with torch.set_grad_enabled(grad):out=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False)
 finally:handle.remove()
 assert len(cache)==1
 a=cache[0];del out
 assert a.shape[:2]==b['input_ids'].shape
 return a
def mask_batch(b,records):
 result=torch.zeros_like(b['input_ids'],dtype=torch.bool)
 for row,doc in enumerate(b['index'].tolist()):
  v=records[doc];n=len(v['mask']);assert b['input_ids'][row,:n].tolist()==v['input_ids']
  result[row,:n]=torch.tensor(v['mask'],device='cuda:0',dtype=torch.bool)
 assert result.shape==b['input_ids'].shape and not (result&~eligible_positions(b)).any()
 return result
def activation_loss(a,b,mask):
 assert a.shape==b.shape and a.shape[:2]==mask.shape
 per=(a.float()-b.float()).square().mean(-1);sums=(per*mask).sum(-1);counts=mask.sum(-1)
 assert sums.shape==counts.shape==(a.shape[0],)
 expected=(sums/counts.clamp_min(1))[counts>0].mean() if (counts>0).any() else sums.sum()*0
 loss=RMU.compute_activation_loss(None,a,b,mask)
 torch.testing.assert_close(loss.detach().float(),expected.detach(),rtol=1e-5,atol=1e-5)
 assert torch.isfinite(loss)
 return loss

@torch.inference_mode()
def diagnostics(model,tokenizer,destination):
 model.eval();samples,masks=inputs();collator=DataCollatorForSupervisedDataset(tokenizer,index='index')
 arrays={};stats={};baseline=np.load(R/'original/token_logprobs.npz') if (R/'original/token_logprobs.npz').exists() and destination!=R/'original' else None
 for role,records in samples.items():
  total=0.;count=0;example_nll=[];selected_total=0.;selected_count=0
  attr_delta=[];nonattr_delta=[];random_delta=[];nonrandom_delta=[]
  for doc,x in records.items():
   b=batch(records,[doc],collator)
   out=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False)
   scores=-F.cross_entropy(out.logits[:,:-1].float().transpose(1,2),b['labels'][:,1:],ignore_index=-100,reduction='none')
   scores=F.pad(scores,(1,0),value=0.)[0].float().cpu().numpy();del out
   eligible=np.asarray(x['eligible'],dtype=bool);arrays[f'{role}_{doc}']=scores
   total-=float(scores[eligible].sum());count+=int(eligible.sum());example_nll.append(float(-scores[eligible].mean()))
   if role=='forget':
    attr=np.asarray(masks['attribute'][doc]['mask'],dtype=bool);rnd=np.asarray(masks['random_attribute'][doc]['mask'],dtype=bool)
    selected_total-=float(scores[attr].sum());selected_count+=int(attr.sum())
    if baseline is not None:
     diff=scores-baseline[f'{role}_{doc}'];attr_delta.extend(diff[attr].tolist());nonattr_delta.extend(diff[eligible&~attr].tolist());random_delta.extend(diff[rnd].tolist());nonrandom_delta.extend(diff[eligible&~rnd].tolist())
  stats[role]={'token_weighted_answer_nll':total/count,'mean_example_nll':float(np.mean(example_nll)),'eligible_tokens':count,'examples':len(records)}
  if role=='forget':
   stats[role]['common_attribute_selected_nll']=selected_total/selected_count
   stats[role]['common_nonattribute_nll']=(total-selected_total)/(count-selected_count)
   if baseline is not None:stats[role]['logprob_change_from_original']={
    'common_attribute_positions':float(np.mean(attr_delta)),'common_nonattribute_positions':float(np.mean(nonattr_delta)),
    'common_random_positions':float(np.mean(random_delta)),'common_nonrandom_positions':float(np.mean(nonrandom_delta))}
 destination.mkdir(exist_ok=True)
 np.savez_compressed(destination/'token_logprobs.npz',**arrays);save(destination/'text_diagnostics.json',stats)
 return stats

def prepare_qa():
 import pyarrow.parquet as pq
 import pyarrow as pa
 config=read(OLD/'frozen_configs/official_evaluator.json');m=read(OLD/'protocol_manifest.json')
 result={k:copy.deepcopy(config[k]) for k in config if k!='metrics'}
 result['metrics']={};audit={}
 for metric in ['forget_knowmem_ROUGE','retain_knowmem_ROUGE']:
  spec=copy.deepcopy(config['metrics'][metric]);ds=next(iter(spec['datasets'].values()))['args']
  source=Path(ds['hf_args']['data_files']);assert sha(source)==m['dataset']['parquet_hashes']['knowmem/'+source.name]
  table=pq.read_table(source);ids=[i*(len(table)-1)//15 for i in range(16)]
  subset=R/'inputs'/f'{metric}_pilot16.parquet';pq.write_table(table.take(pa.array(ids)),subset)
  ds['hf_args']['data_files']=str(subset);spec['batch_size']=4
  result['metrics'][metric]=spec;audit[metric]={'original_sample_ids':ids,'subset_sha256':sha(subset),'source_sha256':sha(source)}
 save(R/'inputs/qa_subset_manifest.json',audit)
 save(R/'inputs/qa_evaluator.json',result)

@torch.inference_mode()
def qa_eval(model,tokenizer,destination):
 from evals import get_evaluators
 from rouge_score.rouge_scorer import RougeScorer
 config=read(R/'inputs/qa_evaluator.json');config['output_dir']=str(destination/'qa');config['overwrite']=False;config['strict_outputs']=True
 model.eval();model.config.use_cache=True
 template=read(R/'plan.json')['model']['template_args']
 get_evaluators(OmegaConf.create({'muse':config}))['muse'].evaluate(model=model,tokenizer=tokenizer,template_args=template)
 model.config.use_cache=False
 logs=read(destination/'qa/MUSE_EVAL.json');scorer=RougeScorer(['rougeL'],use_stemmer=True);summary={}
 assert set(logs)=={'forget_knowmem_ROUGE','retain_knowmem_ROUGE'}
 for name,d in logs.items():
  rows=d['value_by_index'];assert len(rows)==16
  values=[scorer.score(x['ground_truth'],x['generation'])['rougeL'].fmeasure for x in rows.values()]
  value=float(np.mean(values));assert math.isclose(value,d['agg_value'],abs_tol=1e-12)
  summary[name]={'rougeL_f1':value,'count':16,'independent_score_replay':True}
 save(destination/'qa_pilot_metrics.json',summary)
 return summary

def baseline():
 plan=locked();assert not (R/'original').exists()
 (R/'original').mkdir();prepare_qa()
 model,tok=load();samples,masks=inputs();collator=DataCollatorForSupervisedDataset(tok,index='index')
 stats=diagnostics(model,tok,R/'original')
 refdir=R/'retain_reference';refdir.mkdir()
 order=read(R/'data_order.json');entries=[]
 for n in range(32):
  ids=order['retain_ids'][2*n:2*n+2];b=batch(samples['retain'],ids,collator)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=hidden(model,b,False).detach().cpu()
  f=refdir/f'microbatch_{n:02d}.pt';torch.save(a,f)
  entries.append({'microbatch':n,'ids':ids,'shape':list(a.shape),'path':str(f),'sha256':sha(f),'input_ids_sha256':hashlib.sha256(b['input_ids'].cpu().numpy().tobytes()).hexdigest()})
 save(refdir/'manifest.json',{'target_revision':plan['target_revision'],'reference':'frozen exact initial target layer7 activations, evaluation mode, BF16, exact same microbatch pairs/padding as training','entries':entries,'training_updates':0})
 qa=qa_eval(model,tok,R/'original')
 save(R/'original/completed.json',{'phase':'pilot diagnostics only','training_updates':0,'text':stats,'qa':qa,'reference_manifest_sha256':sha(refdir/'manifest.json')})
 print('ORIGINAL_DIAGNOSTICS_COMPLETE',json.dumps({'text':stats,'qa':qa}),flush=True)

def train(arm):
 plan=locked();assert arm in ['attribute','random_attribute']
 assert (R/'original/completed.json').exists();dest=R/arm;assert not dest.exists();dest.mkdir()
 set_seed(0);model,tok=load();samples,masks=inputs();collator=DataCollatorForSupervisedDataset(tok,index='index')
 order=read(R/'data_order.json');ref=read(R/'retain_reference/manifest.json')
 assert sha(R/'retain_reference/manifest.json')==read(R/'original/completed.json')['reference_manifest_sha256']
 # Verify frozen cache against a fresh independent initialization, before updates.
 b=batch(samples['retain'],order['retain_ids'][:2],collator)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=hidden(model,b,False).cpu()
 cached=torch.load(ref['entries'][0]['path'],map_location='cpu',weights_only=True)
 torch.testing.assert_close(a,cached,rtol=0,atol=0);del a,cached,b
 regex=re.compile(plan['method_args']['trainable_params_regex'][0]);params={}
 versions={n:p._version for n,p in model.named_parameters()}
 for n,p in model.named_parameters():p.requires_grad_(bool(regex.fullmatch(n)))
 params={n:p for n,p in model.named_parameters() if p.requires_grad}
 assert params and {int(n.split('.')[2]) for n in params}=={5,6,7}
 initial={n:p.detach().cpu().clone() for n,p in params.items()}
 from bitsandbytes.optim import PagedAdamW32bit
 optimizer=PagedAdamW32bit(list(params.values()),lr=1e-5,weight_decay=0.)
 assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in params.values()}
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model.config.use_cache=False;model.train()
 control=read(R/'inputs/control_vector.json');vec=torch.tensor(control['values'],dtype=torch.float32).reshape(control['shape'])
 assert hashlib.sha256(vec.numpy().tobytes()).hexdigest()==read(OLD/'protocol_manifest.json')['control_vector']['float32_sha256']
 assert math.isclose(vec.norm().item(),1.,rel_tol=1e-6)
 save(dest/'actual_config.json',{'model':plan['model'],'seed':0,'max_steps':4,'learning_rate':1e-5,'alpha':3,'gamma':1,'steering_coeff':1,'module':'model.layers.7','trainable_names':list(params),'trainable_parameter_count':sum(p.numel() for p in params.values()),'microbatch':2,'accumulation':8,'effective_batch':16,'world_size':1,'optimizer':'PagedAdamW32bit','betas':[0.9,0.999],'eps':1e-8,'schedule':'constant','warmup':0,'weight_decay':0,'max_grad_norm':1.,'bf16':True,'gradient_checkpointing':'non-reentrant to preserve layer-hook autograd','reference_representation':'cached immutable exact initial-target layer7 activations for the same microbatches; avoids a second resident 7B copy','outside_optimizer_parameters_frozen':True,'layout_equivalence_claimed':False,'source_sha256':sha(Path(__file__)),'mask_sha256':sha(R/f'masks_{arm}.jsonl'),'frozen_inputs_sha256':sha(R/'FROZEN.json'),'reference_manifest_sha256':sha(R/'retain_reference/manifest.json')})
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();logs=[];seen=[];retain_seen=[]
 for step in range(4):
  optimizer.zero_grad(set_to_none=True);micro=[]
  for accum in range(8):
   n=step*8+accum;fids=order['forget_ids'][2*n:2*n+2];rids=order['retain_ids'][2*n:2*n+2]
   fb=batch(samples['forget'],fids,collator);rb=batch(samples['retain'],rids,collator)
   mask=mask_batch(fb,masks[arm]);entry=ref['entries'][n]
   assert entry['ids']==rids and sha(Path(entry['path']))==entry['sha256']
   assert hashlib.sha256(rb['input_ids'].cpu().numpy().tobytes()).hexdigest()==entry['input_ids_sha256']
   target=torch.load(entry['path'],map_location='cpu',weights_only=True).to('cuda:0')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    af=hidden(model,fb,True);assert af.requires_grad
    fl=activation_loss(af,vec.to(af.device,dtype=af.dtype).expand_as(af),mask)
    ar=hidden(model,rb,True);rl=activation_loss(ar,target,eligible_positions(rb))
    loss=fl+3*rl
   assert torch.isfinite(loss) and fl.ndim==rl.ndim==0
   (loss/8).backward()
   assert all(p.grad is None for name,p in model.named_parameters() if name not in params)
   micro.append({'microbatch':n,'forget_ids':fids,'retain_ids':rids,'forget_loss':float(fl.detach()),'retain_loss':float(rl.detach()),'total_loss':float(loss.detach())})
   seen.extend(fids);retain_seen.extend(rids)
   del af,ar,fl,rl,loss,fb,rb,mask,target
  norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
  assert torch.isfinite(norm) and norm.item()>0
  optimizer.step();torch.cuda.synchronize()
  row={'optimizer_step':step+1,'gradient_norm_preclip':float(norm),'forget_loss':float(np.mean([x['forget_loss'] for x in micro])),'retain_loss':float(np.mean([x['retain_loss'] for x in micro])),'total_loss':float(np.mean([x['total_loss'] for x in micro])),'sample_exposures':len(seen),'microbatches':micro}
  logs.append(row);save(dest/'training_log.json',logs);print('PILOT_UPDATE',arm,json.dumps({k:v for k,v in row.items() if k!='microbatches'}),flush=True)
 torch.cuda.synchronize();runtime=time.perf_counter()-start
 assert seen==order['forget_ids'] and retain_seen==order['retain_ids'] and len(seen)==64
 unchanged=[n for n,p in model.named_parameters() if n not in params and p._version!=versions[n]];assert not unchanged,unchanged
 changes={}
 for n,p in params.items():
  after=p.detach().cpu();delta=after.float()-initial[n].float()
  changes[n]={'changed_elements':int(torch.count_nonzero(delta)),'delta_l2':float(delta.square().sum().sqrt()),'max_abs_delta':float(delta.abs().max())}
 assert sum(v['changed_elements'] for v in changes.values())>0
 from safetensors.torch import save_file
 checkpoint=dest/'checkpoint-4';checkpoint.mkdir()
 saved={n:p.detach().cpu().contiguous() for n,p in params.items()}
 save_file(saved,str(checkpoint/'updated_layers.safetensors'))
 save(checkpoint/'checkpoint_metadata.json',{'kind':'partial state: updated layers5-7; load into the pinned original target with strict=False','base_model':plan['model'],'optimizer_steps':4,'seed':0,'weights_sha256':sha(checkpoint/'updated_layers.safetensors'),'tensor_names':list(saved)})
 from safetensors import safe_open
 with safe_open(str(checkpoint/'updated_layers.safetensors'),framework='pt',device='cpu') as f:
  assert set(f.keys())==set(params)
  assert all(torch.equal(f.get_tensor(n),saved[n]) for n in saved)
 save(dest/'parameter_update_audit.json',{'updated_parameter_changes':changes,'unselected_parameter_versions_unchanged':True,'optimizer_membership_exact':True,'only_layers_5_7_updated':True,'saved_state_matches_model':True})
 del saved,initial,optimizer
 model.gradient_checkpointing_disable();model.eval()
 stats=diagnostics(model,tok,dest);qa=qa_eval(model,tok,dest)
 save(dest/'completed.json',{'arm':arm,'phase':'pilot diagnostics only','seed':0,'optimizer_updates':4,'sample_exposures':64,'unique_forget_samples':32,'forget_order_sha256':hashlib.sha256(json.dumps(seen).encode()).hexdigest(),'retain_order_sha256':hashlib.sha256(json.dumps(retain_seen).encode()).hexdigest(),'training_seconds':runtime,'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'gpu':torch.cuda.get_device_name(0),'finite_losses_and_gradients':True,'reference_frozen':True,'partial_checkpoint_preserved':str(checkpoint/'updated_layers.safetensors'),'text':stats,'qa':qa})
 print('PILOT_ARM_COMPLETE',arm,json.dumps({'text':stats,'qa':qa}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['baseline','train']);parser.add_argument('--arm',choices=['attribute','random_attribute']);args=parser.parse_args()
 if args.phase=='baseline':baseline()
 else:assert args.arm;train(args.arm)
