import sys, os, json, hashlib, copy, shutil, datetime, subprocess
from pathlib import Path
P=Path('/workspace/open-unlearning')
OLD=P/'saves/experiments/muse_books_rmu_multiseed_v1'
SRC=OLD/'source'
sys.path[:0]=[str(SRC/'src')]
import torch
import numpy as np
from omegaconf import OmegaConf
from model import get_tokenizer
from data import get_datasets
from data.collators import DataCollatorForSupervisedDataset
from trainer.masking import eligible_positions
from trainer.unlearn.rmu import RMU
R=P/'saves/experiments/muse_books_attribute_pilot_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not R.exists(), 'Never overwrite a pilot directory'
R.mkdir();(R/'code').mkdir();(R/'inputs').mkdir();(R/'logs').mkdir()
shutil.copy2(__file__,R/'code/prepare_pilot.py')
c=json.loads((OLD/'frozen_configs/adaptive.json').read_text())
m=json.loads((OLD/'protocol_manifest.json').read_text())
for name in ['src/trainer/unlearn/rmu.py','src/trainer/masking.py','src/data/pretraining.py','src/data/utils.py','src/data/collators.py']:
 assert sha(SRC/name)==m['source_sha256'][name],name
 dest=R/'code/original'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/name,dest)
t=m['training'];assert (t['alpha'],t['steering_coeff'],t['learning_rate'])==(3,1,1e-5)
assert t['trainable_params_regex']==[r'model\.layers\.(5|6|7)\..*']
assert t['module_regex']==r'model\.layers\.7' and t['max_sequence_length_parameter']==2048
assert t['lr_scheduler_type']=='constant' and t['weight_decay']==0 and t['warmup_steps']==0
assert t['global_effective_batch']==16 and t['micro_batch']==2
cfg=OmegaConf.create(c['model']);tokenizer=get_tokenizer(cfg.tokenizer_args)
assert tokenizer.is_fast
datasets={role:get_datasets(OmegaConf.create(c['data'][role]),tokenizer=tokenizer,template_args=cfg.template_args) for role in ['forget','retain']}
data_root=Path(c['data']['forget']['MUSE_forget']['args']['hf_args']['data_files']).parent
for split in ['forget','retain1']:
 f=data_root/(split+'-00000-of-00001.parquet');assert sha(f)==m['dataset']['parquet_hashes']['raw/'+f.name]
ids={role:[i*(len(datasets[role])-1)//(n-1) for i in range(n)] for role,n in [('forget',32),('retain',16)]}
assert len(set(ids['forget']))==32 and len(set(ids['retain']))==16
base={}
with (OLD/'frozen_inputs/base.jsonl').open() as f:
 for line in f:
  d=json.loads(line)
  if d['doc_id'] in ids['forget']:base[d['doc_id']]=d
records={}
for role in ['forget','retain']:
 records[role]=[]
 for doc in ids[role]:
  item=datasets[role][doc];text=datasets[role].chunks[doc]
  enc=tokenizer(text,add_special_tokens=True,return_offsets_mapping=True)
  tokens=item['input_ids'].tolist();labels=item['labels'].tolist();attention=item['attention_mask'].tolist()
  assert enc['input_ids'][:len(tokens)]==tokens
  offsets=[list(x) for x in enc['offset_mapping'][:len(tokens)]]
  valid=eligible_positions({k:v[None] for k,v in item.items() if k!='index'})[0].tolist()
  if role=='forget':
   assert base[doc]['input_ids']==tokens
   if 'labels' in base[doc]:assert base[doc]['labels']==labels
  assert tokens[0]==tokenizer.bos_token_id and not valid[0] and len(tokens)<=2049
  records[role].append({'doc_id':doc,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
   'input_ids':tokens,'labels':labels,'attention_mask':attention,'eligible':valid,'offsets':offsets,
   'visible_text_end':max(x[1] for x in offsets),'input_ids_sha256':hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()})
 with (R/'inputs'/f'{role}_samples.jsonl').open('w') as f:
  for d in records[role]:f.write(json.dumps(d,ensure_ascii=False)+'\n')
prompt='''Annotate the provided MUSE Books forget TRAINING text blocks only. Do not consult any evaluation question, answer, or model result. For each block, select zero to four unambiguous, concrete factual spans (events, relationships, locations, times or attributes). This is a non-exhaustive pilot annotation, not a complete inventory of all facts. Prefer the smallest literal contiguous phrase carrying the proposition, predicate/relationship and factual value; include the subject when needed to make the relation clear. A bare proper name is not an attribute. Do not select a whole sentence merely because it contains a name. Avoid unresolved pronouns, speculation and rhetorical dialogue where factual attribution is unclear. Spans must lie inside visible_text_end. Return exact source substrings and a brief category/reason. Empty annotations are allowed. Character offsets and unique source matches will be checked programmatically. Do not target a prescribed token density. Semantic checks by the generating assistant are model self-checks, never independent human review.'''
(R/'annotation_prompt.txt').write_text(prompt+'\n')
save(R/'annotation_provenance.json',{'generator_tool':'Codex assistant in the current desktop conversation','generation_mode':'Spans selected by the current assistant from displayed original training blocks; no additional model agent or remote Codex exec',
 'actual_model_id':None,'actual_model_version':None,'reasoning_effort':None,
 'model_information_note':'Runtime model identifier/version/effort are not exposed to this assistant; do not infer gpt-6.1-sol from older TOFU sessions.',
 'human_review_completed':False,'semantic_review':'generator self-check only','alignment_review':'pending programmatic literal/offset/token audit',
 'evaluation_answers_used_for_annotation':False,'maximum_spans_per_block':4})
control=OLD/'frozen_inputs/control_vector.json';assert sha(control)==m['control_vector']['file_sha256'];shutil.copy2(control,R/'inputs/control_vector.json')
order=[];retain_order=[]
g=torch.Generator().manual_seed(0);rg=torch.Generator().manual_seed(2231)
for epoch in range(2):
 order += [ids['forget'][i] for i in torch.randperm(32,generator=g).tolist()]
 retain_order += [ids['retain'][i] for i in torch.randint(16,(32,),generator=rg).tolist()]
save(R/'data_order.json',{'forget_ids':order,'retain_ids':retain_order,'training_seed':0,'retain_order_seed':2231,'sample_exposures':64,'effective_batch':16,'microbatch':2,'accumulation':8})
plan={'phase':'MUSE Books Attribute-only vs Random-Attribute seed0 bounded pilot diagnostics','max_optimizer_updates_per_arm':4,'training_seeds':[0],
 'arms':['attribute','random_attribute'],'forget_count':32,'retain_count':16,'selection_rule':'32 forget and 16 retain IDs floor(i*(N-1)/(n-1)) across actual training chunks, in ascending order; independent of annotations/evaluation',
 'sample_ids':ids,'full_dataset_counts':{k:len(v) for k,v in datasets.items()},'original_protocol_training':t,
 'pilot_layout':{'world_size':1,'microbatch':2,'gradient_accumulation':8,'effective_batch':16,'numerically_equivalent_to_original_two_gpu_layout':False},
 'mask_seed':20261122,'maximum_annotation_spans_per_sample':4,'max_length_parameter':2048,'max_actual_input_tokens':max(len(x['input_ids']) for x in records['forget']+records['retain']),
 'bos_token_id':tokenizer.bos_token_id,'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id,
 'tokenizer_revision':m['models']['tokenizer']['revision'],'target_revision':m['models']['target']['revision'],'dataset_revision':m['dataset']['revision'],
 'model':c['model'],'method_args':c['trainer']['method_args'],'original_protocol_sha256':sha(OLD/'protocol_manifest.json'),
 'source_sha256':{n:sha(SRC/n) for n in ['src/trainer/unlearn/rmu.py','src/trainer/masking.py','src/data/pretraining.py','src/data/utils.py','src/data/collators.py']},
 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),'training_or_eval_started':False,
 'checkpoint_retention':'Explicit current user instruction: preserve all new and old model checkpoints; no cleanup this pilot.',
 'qa_protocol':'Optional exactly 16 evenly spaced forget QA and 16 retain QA, official configuration/scorer, only prepared after annotation/mask freeze. No ES/privacy evaluation.'}
save(R/'plan.json',plan)
save(R/'preflight_cpu.json',{'preprocessing_matches_frozen_formal_inputs':True,'original_corrected_rmu_sha_verified':True,'original_data_file_hashes_verified':True,'selected_counts':{k:len(v) for k,v in records.items()},'no_qa_data_loaded':True})
print(json.dumps({'root':str(R),'dataset_counts':plan['full_dataset_counts'],'selected_ids':ids,'lengths':{k:[len(x['input_ids']) for x in v] for k,v in records.items()},'tokenizer_fast':True},indent=2))
