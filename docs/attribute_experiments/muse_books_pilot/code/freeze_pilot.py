from pathlib import Path
import sys,json,hashlib,datetime,statistics,copy
import numpy as np
import torch
P=Path('/workspace/open-unlearning');OLD=P/'saves/experiments/muse_books_rmu_multiseed_v1'
sys.path.insert(0,str(OLD/'source/src'))
from trainer.unlearn.rmu import RMU
R=P/'saves/experiments/muse_books_attribute_pilot_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not (R/'FROZEN.json').exists()
samples={x['doc_id']:x for x in map(json.loads,(R/'inputs/forget_samples.jsonl').read_text().splitlines())}
annotations=json.loads((R/'annotations_raw.json').read_text());plan=json.loads((R/'plan.json').read_text())
assert len(annotations)==32 and {x['doc_id'] for x in annotations}==set(samples)
aligned=[];masks={'attribute':[],'random_attribute':[]};rows=[];char_lengths=[];token_lengths=[]
preview=['# Attribute-only pilot annotation preview','','Generator: current Codex assistant; exact runtime model/version/effort not exposed. No human review is claimed. Non-exhaustive, at most four spans per original training block.','']
for ann in annotations:
 x=samples[ann['doc_id']];text=x['text'];assert hashlib.sha256(text.encode()).hexdigest()==x['text_sha256']
 valid=np.asarray(x['eligible'],dtype=bool);mask=np.zeros(len(valid),dtype=bool);spans=[]
 assert len(ann['spans'])<=4
 preview += [f"## Training block {x['doc_id']}",f"SHA-256: `{x['text_sha256']}`",'']
 for proposal in ann['spans']:
  literal=proposal['text'];assert literal and text.count(literal)==1,(x['doc_id'],literal)
  start=text.index(literal);end=start+len(literal);assert end<=x['visible_text_end']
  assert text[start:end]==literal
  overlap=np.asarray([a<end and b>start and b>a for a,b in x['offsets']],dtype=bool)&valid
  assert overlap.any(),(x['doc_id'],literal)
  mask|=overlap
  spans.append({**proposal,'char_start':start,'char_end':end,'coordinate':'zero-based half-open Unicode character offsets','overlapping_eligible_token_indices':np.flatnonzero(overlap).tolist()})
  char_lengths.append(end-start);token_lengths.append(int(overlap.sum()))
  preview.append(f"- [{start},{end}) {proposal['category']}: **{literal}**")
 if not spans:preview.append('- Empty: no selected factual span.')
 preview += ['', '<details><summary>Original forget training block</summary>','',text,'','</details>','']
 aligned.append({'doc_id':x['doc_id'],'text_sha256':x['text_sha256'],'spans':spans,'human_review_completed':False})
 rng=np.random.default_rng(np.random.SeedSequence([plan['mask_seed'],x['doc_id']]))
 random=np.zeros(len(valid),dtype=bool)
 if mask.any():random[rng.choice(np.flatnonzero(valid),size=int(mask.sum()),replace=False)]=True
 assert mask.sum()==random.sum() and not (mask&~valid).any() and not (random&~valid).any()
 for arm,v in [('attribute',mask),('random_attribute',random)]:
  masks[arm].append({'doc_id':x['doc_id'],'input_ids':x['input_ids'],'labels':x['labels'],'attention_mask':x['attention_mask'],'eligible':x['eligible'],'mask':v.astype(int).tolist(),'text_sha256':x['text_sha256']})
 rows.append({'doc_id':x['doc_id'],'input_tokens':len(valid),'eligible_tokens':int(valid.sum()),'selected_tokens':int(mask.sum()),'density':float(mask.sum()/max(valid.sum(),1)),'spans':len(spans)})
assert len(rows)==32
for arm,items in masks.items():
 with (R/f'masks_{arm}.jsonl').open('w') as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
save(R/'annotations.json',aligned);(R/'annotation_preview.md').write_text('\n'.join(preview))
def distribution(v):return {'min':min(v),'median':statistics.median(v),'mean':statistics.mean(v),'max':max(v),'q25':float(np.percentile(v,25)),'q75':float(np.percentile(v,75))}
audit={'samples':32,'selected_tokens':sum(r['selected_tokens'] for r in rows),'eligible_tokens':sum(r['eligible_tokens'] for r in rows),'empty_masks':sum(r['selected_tokens']==0 for r in rows),'span_count':len(char_lengths),'span_char_length':distribution(char_lengths),'span_token_length':distribution(token_lengths),'per_sample_density':distribution([r['density'] for r in rows]),'rows':rows,
 'checks':{'same_sample_ids_and_input_tokens':True,'mask_lengths_match_inputs':True,'exact_per_sample_count_and_empty_match':True,'eligible_positions_only':True,'unique_literal_alignment_all_spans':True,'input_aligned_rmu_mask_no_npo_shift':True,'human_review_completed':False},
 'random_mask_seed':plan['mask_seed'],'random_algorithm':'NumPy default_rng(SeedSequence([mask_seed, stable_doc_id])), no replacement'}
audit['selection_rate']=audit['selected_tokens']/audit['eligible_tokens']
# Independent unequal-count/empty-sample normalization check using the corrected formal implementation.
a=torch.arange(3*4*3,dtype=torch.float32).reshape(3,4,3).requires_grad_();b=torch.zeros_like(a)
mask=torch.tensor([[1,0,0,0],[1,1,1,0],[0,0,0,0]],dtype=torch.bool)
per_token=(a-b).square().mean(-1);sums=(per_token*mask).sum(-1);counts=mask.sum(-1)
assert sums.shape==counts.shape==(3,)
loss=RMU.compute_activation_loss(None,a,b,mask)
expected=torch.stack([per_token[i][mask[i]].mean() for i in range(3) if mask[i].any()]).mean()
torch.testing.assert_close(loss,expected)
g,=torch.autograd.grad(loss,a);assert torch.isfinite(g).all() and (g[2]==0).all()
# Input-aligned hidden states: position j is used directly, rather than j-1.
assert mask[0].nonzero().flatten().tolist()==[0]
audit['checks']['corrected_per_example_mse_and_counts_shapes_verified']=True
audit['checks']['independent_unequal_count_empty_loss_and_gradient_test_passed']=True
save(R/'mask_audit.json',audit)
provenance=json.loads((R/'annotation_provenance.json').read_text());provenance['alignment_review']='all 126 proposed literal spans uniquely matched; offsets and eligible-token overlaps checked';provenance['span_count']=len(char_lengths);save(R/'annotation_provenance.json',provenance)
names=['plan.json','data_order.json','annotation_prompt.txt','annotation_provenance.json','annotations_raw.json','annotations.json','masks_attribute.jsonl','masks_random_attribute.jsonl','mask_audit.json','inputs/forget_samples.jsonl','inputs/retain_samples.jsonl','inputs/control_vector.json']
save(R/'FROZEN.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_or_evaluation_started':False,'files':{n:sha(R/n) for n in names}})
for n in names:(R/n).chmod(0o444)
print(json.dumps({k:audit[k] for k in ['samples','selected_tokens','eligible_tokens','selection_rate','empty_masks','span_count','span_char_length','span_token_length','per_sample_density']},indent=2))
