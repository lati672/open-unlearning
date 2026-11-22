"""CPU-only replay: python code/replay_scores.py [experiment_root]. No model needed."""
import json,sys,math,csv
from pathlib import Path
from statistics import mean
import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from scipy.stats import hmean,ks_2samp
from sklearn.metrics import roc_auc_score
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
ref=read(R/'manifests/retain_only_reference_TOFU_EVAL.json')
counts={'official':0,'question_conditions':0,'question_rows':0}
raw_conditions={}
for p in (R/'evaluations/reproduction').rglob('official/TOFU_EVAL.json'):
 logs=read(p);cfg=read(p.parent/'evaluation_provenance.json')['resolved_eval_config'];actual=read(p.parent/'official_metrics.json')
 vals=lambda name,key:[x[key] for x in logs[name]['value_by_index'].values()]
 def agg(name,spec):
  if spec['handler'] in ['probability','probability_w_options']:return mean(vals(name,'prob'))
  if spec['handler']=='rouge':return mean(vals(name,spec['rouge_type']))
  if spec['handler']=='truth_ratio':
   v=vals(name,'score');return mean(max(0,1-x) if spec['aggregator']=='true_better' else min(x,1/(x+1e-10)) for x in v)
  raise ValueError(spec['handler'])
 scores=[[x['score'] for x in logs['mia_min_k'][split]['value_by_index'].values()] for split in ['forget','holdout']]
 auc=roc_auc_score([0]*len(scores[0])+[1]*len(scores[1]),scores[0]+scores[1]);ra=ref['mia_min_k']['agg_value']
 tr=vals('forget_truth_ratio','score');rt=[x['score'] for x in ref['forget_truth_ratio']['value_by_index'].values()]
 metrics={'ES':mean(vals('extraction_strength','score')),'forget_probability':mean(vals('forget_Q_A_Prob','prob')),'forget_rouge':mean(vals('forget_Q_A_ROUGE','rougeL_recall')),'forget_ratio':agg('forget_truth_ratio',cfg['metrics']['forget_truth_ratio']),'forget_quality':float(ks_2samp(tr,rt).pvalue),'utility':float(hmean([agg(n,s) for n,s in cfg['metrics']['model_utility']['pre_compute'].items()])),'privleak':((1-auc)-(1-ra))/(1-ra+1e-10)*100}
 assert all(math.isclose(v,actual[k],rel_tol=1e-10,abs_tol=1e-14) for k,v in metrics.items()),p
 counts['official']+=1
scorer=RougeScorer(['rougeL'],use_stemmer=True)
for cp in (R/'evaluations').rglob('questions/completed.json'):
 for style in ['original','ordinary','substantial']:
  d=read(cp.parent/f'{style}.json');assert [x['original_id'] for x in d['rows']]==list(range(200))
  method='OriginalTarget' if d['aggregation']=='target' else d['method']
  seed='' if d['seed'] is None else str(d['seed'])
  raw_conditions[d['aggregation'],method,seed,style]=d['rows']
  for x in d['rows']:
   assert math.isclose(math.exp(sum(x['token_logprobs'])/x['token_count']),x['prob'],rel_tol=2e-5,abs_tol=1e-7)
   assert scorer.score(x['ground_truth'],x['generation'])['rougeL'].recall==x['rougeL_recall']
  assert math.isclose(mean(x['prob'] for x in d['rows']),d['summary']['F_Prob'],rel_tol=1e-12)
  assert math.isclose(mean(x['rougeL_recall'] for x in d['rows']),d['summary']['F_ROUGE'],rel_tol=1e-12)
  counts['question_conditions']+=1;counts['question_rows']+=len(d['rows'])
assert counts==dict(official=48,question_conditions=39,question_rows=7800),counts
csvrows=lambda n:list(csv.DictReader((R/'paraphrases/analysis'/n).open()))
masks=[json.loads(s) for s in (R/'manifests/frozen_inputs/AttributeOnly.jsonl').read_text().splitlines()]
empty={p['doc_id'] for p in masks if not any(p['mask'])}
groups={'all200':set(range(200)),'nonempty185':set(range(200))-empty,'empty15':empty,'equivalence_sensitivity198':set(range(200))-{117,131},'postfreeze_text_quality_sensitivity195':set(range(200))-{117,131,124,128,151}}
metric_index={}
for p in csvrows('metrics_by_seed.csv'):
 agg,method,seed,style=p['aggregation'],p['method'],p['seed'],p['style'];key='prob' if p['metric']=='F-Prob' else 'rougeL_recall';ids=groups[p['group']]
 if style=='paraphrase_pair_average':v=mean(.5*(raw_conditions[agg,method,seed,'ordinary'][i][key]+raw_conditions[agg,method,seed,'substantial'][i][key]) for i in ids)
 else:v=mean(raw_conditions[agg,method,seed,style][i][key] for i in ids)
 assert len(ids)==int(p['n_original_ids']) and math.isclose(v,float(p['value']),rel_tol=1e-12,abs_tol=1e-14)
 metric_index[agg,method,seed,p['group'],style,p['metric']]=v
for p in csvrows('mean_sd.csv'):
 vals=[metric_index[p['aggregation'],p['method'],str(s),p['group'],p['style'],p['metric']] for s in range(3)]
 assert all(math.isclose(v,float(p[f'seed{s}']),rel_tol=1e-12,abs_tol=1e-14) for s,v in enumerate(vals))
 assert math.isclose(float(np.mean(vals)),float(p['mean']),rel_tol=1e-12,abs_tol=1e-14) and math.isclose(float(np.std(vals,ddof=1)),float(p['sample_sd']),rel_tol=1e-12,abs_tol=1e-14)
for p in csvrows('paired_method_deltas.csv'):
 vals=[metric_index[p['aggregation'],'AttributeOnly',str(s),p['group'],p['style'],p['metric']]-metric_index[p['aggregation'],'RandomAttribute',str(s),p['group'],p['style'],p['metric']] for s in range(3)]
 assert all(math.isclose(v,float(p[f'seed{s}']),rel_tol=1e-12,abs_tol=1e-14) for s,v in enumerate(vals))
 assert math.isclose(float(np.mean(vals)),float(p['mean']),rel_tol=1e-12,abs_tol=1e-14) and math.isclose(float(np.std(vals,ddof=1)),float(p['sample_sd']),rel_tol=1e-12,abs_tol=1e-14)
print(json.dumps({'passed':True,**counts,'all_subgroup_means_recomputed_from_rows':True,'three_seed_sample_sd_replayed':True,'pair_by_seed_before_aggregation_verified':True}))
