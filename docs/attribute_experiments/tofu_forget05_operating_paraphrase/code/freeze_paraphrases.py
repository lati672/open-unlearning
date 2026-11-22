import json,csv,hashlib,re,os
from pathlib import Path
from datetime import datetime,timezone
from datasets import load_dataset
R=Path('/workspace/open-unlearning/saves/experiments/tofu_forget05_attribute_operating_paraphrase_v1')
P=R/'paraphrases'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert not (P/'FROZEN.json').exists()
qs=[json.loads(x) for x in (P/'original_questions_only.jsonl').read_text().splitlines()]
raw=list(csv.reader((P/'generation_raw.tsv').read_text().splitlines(),delimiter='\t'))
assert len(qs)==len(raw)==200
ds=load_dataset('locuslab/TOFU','forget05',revision='324592d84ae4f482ac7249b9285c2ecdb53e3a68',split='train')
norm=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
rows=[];checks=[];uncertain=[];seen={}
for i,(q,r) in enumerate(zip(qs,raw)):
 assert int(r[0])==q['original_id']==i and len(r)==3
 assert q['question']==ds[i]['question']
 for style,p in zip(['ordinary','substantial'],r[1:]):
  assert p and norm(p)!=norm(q['question'])
  key=norm(p);duplicate=seen.get(key);assert duplicate is None,(i,duplicate)
  seen[key]=[i,style]
  # Full-answer leakage detection, excluding material already present in source question.
  answer=norm(ds[i]['answer']);new_answer=bool(len(answer.split())>=3 and answer in norm(p) and answer not in norm(q['question']))
  assert not new_answer,(i,style,'new full target answer in rewrite')
  quotes=re.findall(r'"([^"\n]+)"',q['question'])
  assert all(norm(x) in norm(p) for x in quotes),(i,style,'missing quoted constraint')
  note=None
  if i==117:note='Source uses maledictions; intended meaning unclear. Term retained verbatim without repair. Excluded from equivalence-confirmed sensitivity analysis, retained in all-200 primary evaluation.'
  if i==131:note='Source phrase place her character developments is awkward/ambiguous. Conservative literal paraphrases; excluded from equivalence-confirmed sensitivity analysis, retained in all-200 primary evaluation.'
  check=dict(original_id=i,style=style,semantic_checker='generating Codex assistant self-check; not independent human review',entities_constraints_checked=True,no_added_facts_or_task_checked=True,automatic_full_answer_leak=False,duplicate=False,source_ambiguity=note,equivalence_status='uncertain-source-meaning' if note else 'AI-self-check-passed')
  checks.append(check)
  if note:uncertain.append(check)
  rows.append(dict(original_id=i,style=style,original_question=q['question'],question=p,equivalence_status=check['equivalence_status']))
(P/'questions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
save(P/'quality_checks.json',dict(human_review_completed=False,semantic_review='generator self-check only',automatic_checks=['ID alignment','duplicates','full target-answer leakage introduced beyond source','quoted constraints'],primary_all_original_ids=200,primary_paraphrases=400,equivalence_confirmed_by_AI_original_ids=198,equivalence_confirmed_by_AI_paraphrases=396,excluded_from_primary=0,uncertain_items=uncertain,per_item=checks))
save(P/'generation_provenance.json',dict(tool='Codex assistant in current desktop conversation',model_id='unknown',model_version='unknown',reasoning_effort='unknown',temperature='unknown',sampling_settings='unknown',generation_mode='assistant produced TSV from original questions only; no additional model agent',inputs=['original_id','question'],answers_spans_masks_model_outputs_provided=False,post_generation_programmatic_answer_leak_check=True,human_review_completed=False,ambiguous_source_ids=[117,131],source_quirks='Fictitious awards/books requested by original questions retained as fictitious; ambiguous numeric date 08/09/1956 retained unchanged.'))
files=['original_questions_only.jsonl','generation_raw.tsv','generation_prompt.txt','questions.jsonl','quality_checks.json','generation_provenance.json']
save(P/'FROZEN.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),before_any_test_model_evaluation=True,files_sha256={n:sha(P/n) for n in files},primary_original_ids=200,primary_questions_per_style=200,uncertainty_sensitivity_original_ids=198,change_policy='No question edits based on model performance; all200 primary plus198 source-equivalence sensitivity.'))
for n in files+['FROZEN.json']:os.chmod(P/n,0o444)
print(json.dumps(dict(frozen=True,primary_questions=400,uncertain_original_ids=[117,131],human_review=False,sha256=sha(P/'FROZEN.json'))))
