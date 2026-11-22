import json,sys,csv,hashlib,math,importlib.util
from pathlib import Path
from statistics import mean,stdev
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];REPO=Path('/workspace/open-unlearning')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=importlib.util.spec_from_file_location('existing_matching',Path(__file__).with_name('appendix_c_matching.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
METHODS=('AttributeOnly','RandomAttribute');METRICS=('F-Prob','Utility','ES','F-ROUGE','absolute PrivLeak')
def summary(v):
 vals=[x for x in v if x is not None]
 return dict(**{f'seed{s}':x for s,x in enumerate(v)},mean=mean(vals) if vals else None,sample_sd=stdev(vals) if len(vals)>1 else None,n_seeds=len(vals),all_three_present=len(vals)==3,directions=['positive' if x>0 else 'negative' if x<0 else 'tie' for x in vals])
def op_analysis(points,label):
 out=R/'operating_points'/label;out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for p in points:
  rows.append(dict(**{k:p[k] for k in ['aggregation','method','seed','step','trajectory']},variant=p['method'],density=1432/7165*100,**{'F-Prob':p['forget_probability'],'Utility':p['utility'],'ES':p['ES'],'F-ROUGE':p['forget_rouge'],'F-Quality':p['forget_quality'],'absolute PrivLeak':abs(p['privleak'])}))
 csvout(out/'raw_points.csv',rows)
 supports=[];matched=[];grids=[];averages=[];near=[];nearsum=[];fronts=[]
 for agg in ['summed','token_mean']:
  fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
  for seed in range(3):
   raw={method:[p for p in rows if p['aggregation']==agg and p['seed']==seed and p['method']==method] for method in METHODS}
   assert all(raw.values())
   curves={method:m.frontier(v) for method,v in raw.items()}
   for method,ps in curves.items():
    fronts.extend(dict(p,frontier=True) for p in ps)
    observed=sorted(raw[method],key=lambda p:p['step']);color='#167d9a' if method==METHODS[0] else '#bc553e'
    axs[seed].scatter([p['Utility'] for p in observed],[p['F-Prob'] for p in observed],color=color,label=method,s=30)
    axs[seed].plot([p['Utility'] for p in observed],[p['F-Prob'] for p in observed],color=color,ls=':',alpha=.5)
    axs[seed].plot([p['Utility'] for p in ps],[p['F-Prob'] for p in ps],color=color,lw=1.3,label='frontier linear segments' if method==METHODS[0] else None)
    for p in observed:
     offset={30:(-12,6),60:(5,-10),90:(-25,15),120:(6,-14)}[p['step']]
     axs[seed].annotate(str(p['step']),(p['Utility'],p['F-Prob']),fontsize=7,xytext=offset,textcoords='offset points',color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.5,alpha=.6))
   axs[seed].set(title=f'Seed {seed}',xlabel='Utility (higher better)',ylabel='F-Prob (lower better)');axs[seed].grid(alpha=.2)
   for mode,(x,y) in {'matched_utility':('Utility','F-Prob'),'matched_fprob':('F-Prob','Utility')}.items():
    base=dict(aggregation=agg,seed=seed,analysis=mode,match_axis=x,outcome=y)
    common=m.intersection([m.bounds(curves[z],x) for z in METHODS]);rawcommon=m.intersection([m.bounds(raw[z],x) for z in METHODS])
    supports.append(dict(**base,kind='frontier',**{z+'_range':list(m.bounds(curves[z],x)) for z in METHODS},common=list(common) if common else None,status='no common support' if common is None else 'point only' if common[0]==common[1] else 'interval'))
    supports.append(dict(**base,kind='raw',**{z+'_range':list(m.bounds(raw[z],x)) for z in METHODS},common=list(rawcommon) if rawcommon else None,status='no common support' if rawcommon is None else 'point only' if rawcommon[0]==rawcommon[1] else 'interval'))
    if common is not None:
     metricset=[v for v in METRICS if v!=x]
     point_only=common[0]==common[1]
     targets=np.array([common[0]]) if point_only else np.linspace(*common,201)
     local=[]
     for j,t in enumerate(targets):
      values={z:m.interpolate(curves[z],x,float(t),metricset) for z in METHODS}
      row=dict(**base,grid_index=j,target=float(t),common_min=common[0],common_max=common[1],comparison_type='matched_point' if point_only else 'interval_interpolation')
      for z,v in values.items():row.update({z+'_'+k:val for k,val in v.items()})
      row.update({'delta_'+k:values[METHODS[0]][k]-values[METHODS[1]][k] for k in metricset})
      local.append(row)
     grids.extend(local)
     if point_only:matched.append(dict(local[0],normalized_position='point_only'))
     else:
      for pos,idx in [(.25,50),(.5,100),(.75,150)]:matched.append(dict(local[idx],normalized_position=pos))
     for k in metricset:
      d=[z['delta_'+k] for z in local]
      averages.append(dict(**base,metric=k,value=mean(d),effect_min=min(d),effect_max=max(d),n_grid_points=len(targets),common_min=common[0],common_max=common[1],comparison_type='matched_point' if point_only else 'common_support_grid_mean',observational_units='one training seed; coordinates are not independent'))
    # Exact original nearest observed rules, all failures retained.
    candidates=[p for p in raw[METHODS[1]] if rawcommon and rawcommon[0]<=p[x]<=rawcommon[1]]
    for caliper in [.005,.01,.02]:
     accepted=[]
     for p in raw[METHODS[0]]:
      status='no common support' if rawcommon is None else 'outside common support' if not rawcommon[0]<=p[x]<=rawcommon[1] else 'no Random point in common support' if not candidates else 'candidate'
      row=dict(**base,caliper=caliper,attribute_step=p['step'],attribute_axis=p[x],common=list(rawcommon) if rawcommon else None)
      if status=='candidate':
       b=min(candidates,key=lambda q:(abs(p[x]-q[x]),q['step'],q['density']));gap=abs(p[x]-b[x]);status='matched' if gap<=caliper else 'outside caliper'
       row.update(random_step=b['step'],random_axis=b[x],absolute_gap=gap,signed_gap=p[x]-b[x])
       if status=='matched':
        row.update({'delta_'+k:p[k]-b[k] for k in METRICS if k!=x});accepted.append(row)
      row['status']=status;near.append(row)
     nearsum.append(dict(**base,caliper=caliper,matched=len(accepted),attribute_total=len(raw[METHODS[0]]),mean_difference=mean(p['delta_'+y] for p in accepted) if accepted else None,positive_pairs=sum(p['delta_'+y]>0 for p in accepted),negative_pairs=sum(p['delta_'+y]<0 for p in accepted)))
  axs[0].legend(fontsize=7);fig.suptitle(f'{label}: {agg}; dots = measured checkpoints, dotted = chronological order')
  fig.savefig(out/f'{agg}_trajectories.png',dpi=160);fig.savefig(out/f'{agg}_trajectories.pdf');plt.close(fig)
 cross=[]
 for agg in ['summed','token_mean']:
  for mode in ['matched_utility','matched_fprob']:
   for k in METRICS:
    values=[next((p['value'] for p in averages if p['aggregation']==agg and p['analysis']==mode and p['metric']==k and p['seed']==seed),None) for seed in range(3)]
    if any(v is not None for v in values):
     types={p['seed']:p['comparison_type'] for p in averages if p['aggregation']==agg and p['analysis']==mode and p['metric']==k}
     cross.append(dict(aggregation=agg,analysis=mode,metric=k,comparison_types_by_seed=types,**summary(values)))
 nearcross=[]
 for agg in ['summed','token_mean']:
  for mode in ['matched_utility','matched_fprob']:
   for caliper in [.005,.01,.02]:
    values=[next(p['mean_difference'] for p in nearsum if p['aggregation']==agg and p['analysis']==mode and p['caliper']==caliper and p['seed']==seed) for seed in range(3)]
    nearcross.append(dict(aggregation=agg,analysis=mode,caliper=caliper,**summary(values)))
 for name,data in [('frontier_points.csv',fronts),('matched_points_and_quartiles.csv',matched),('matched_coordinates.csv',grids),('matched_effect_by_seed.csv',averages),('matched_effect_summary.csv',cross),('nearest_observed_matches.csv',near),('nearest_observed_summary.csv',nearsum),('nearest_cross_seed_summary.csv',nearcross)]:csvout(out/name,data)
 result=dict(grid_mean_summary=cross,common_support=supports,primary_caliper_nearest=[p for p in nearsum if p['caliper']==.01],nearest_cross_seed_summary=nearcross)
 save(out/'common_support.json',supports);save(out/'summary.json',result)
 return result
def para_analysis():
 models=[]
 root=R/'evaluations/original_target/questions';models.append(('target','OriginalTarget',None,root))
 for a in ['summed','token_mean']:
  for method in METHODS:
   for seed in range(3):models.append((a,method,seed,R/'evaluations/reproduction'/a/method/f'seed{seed}/step120/questions'))
 masks=[json.loads(s) for s in (R/'manifests/frozen_inputs/AttributeOnly.jsonl').read_text().splitlines()]
 empty={p['doc_id'] for p in masks if not any(p['mask'])};assert len(empty)==15
 groups={'all200':set(range(200)),'nonempty185':set(range(200))-empty,'empty15':empty,'equivalence_sensitivity198':set(range(200))-{117,131},'postfreeze_text_quality_sensitivity195':set(range(200))-{117,131,124,128,151}}
 results=[];replay=[];question_changes=[];direction_counts=[]
 from rouge_score.rouge_scorer import RougeScorer
 scorer=RougeScorer(['rougeL'],use_stemmer=True)
 for agg,method,seed,path in models:
  dat={style:read(path/f'{style}.json')['rows'] for style in ['original','ordinary','substantial']}
  for i in range(200):
   a=dat['original'][i]
   for style in ['ordinary','substantial']:
    b=dat[style][i]
    assert a['target_answer']==b['target_answer'] and a['ground_truth']==b['ground_truth']
    assert a['target_token_ids']==b['target_token_ids'] and a['token_count']==b['token_count']
  for style,rows in dat.items():
   assert [x['original_id'] for x in rows]==list(range(200))
   for x in rows:
    assert math.isfinite(x['prob']) and math.isfinite(x['rougeL_recall'])
    p=math.exp(sum(x['token_logprobs'])/x['token_count']);assert math.isclose(p,x['prob'],rel_tol=2e-5,abs_tol=1e-7)
    rg=scorer.score(x['ground_truth'],x['generation'])['rougeL'].recall;assert rg==x['rougeL_recall']
   replay.append(dict(aggregation=agg,method=method,seed=seed,style=style,n=200,probability_from_token_logps_passed=True,rouge_replay_passed=True,same_answer_tokens_and_rouge_reference_across_styles=True))
  for group,ids in groups.items():
   for style in ['original','ordinary','substantial','paraphrase_pair_average']:
    for metric,key in [('F-Prob','prob'),('F-ROUGE','rougeL_recall')]:
     values=[dat[style][i][key] if style!='paraphrase_pair_average' else .5*(dat['ordinary'][i][key]+dat['substantial'][i][key]) for i in sorted(ids)]
     results.append(dict(aggregation=agg,method=method,seed=seed,group=group,style=style,metric=metric,value=mean(values),n_original_ids=len(ids)))
  for i in range(200):
   for style in ['ordinary','substantial']:
    question_changes.append(dict(aggregation=agg,method=method,seed=seed,original_id=i,style=style,empty_attribute_mask=i in empty,delta_FProb=dat[style][i]['prob']-dat['original'][i]['prob'],delta_FROUGE=dat[style][i]['rougeL_recall']-dat['original'][i]['rougeL_recall']))
  for style in ['ordinary','substantial']:
   for metric,key in [('F-Prob','prob'),('F-ROUGE','rougeL_recall')]:
    d=[dat[style][i][key]-dat['original'][i][key] for i in range(200)]
    direction_counts.append(dict(aggregation=agg,method=method,seed=seed,style=style,metric=metric,n_original_ids=200,increased=sum(x>1e-12 for x in d),decreased=sum(x< -1e-12 for x in d),tied=sum(abs(x)<=1e-12 for x in d),mean_change=mean(d)))
 def val(a,method,s,g,style,m):return next(p['value'] for p in results if p['aggregation']==a and p['method']==method and p['seed']==s and p['group']==g and p['style']==style and p['metric']==m)
 summaries=[];paired=[];changes=[]
 for agg in ['summed','token_mean']:
  for group in groups:
   for metric in ['F-Prob','F-ROUGE']:
    for style in ['original','ordinary','substantial','paraphrase_pair_average']:
     for method in METHODS:
      summaries.append(dict(aggregation=agg,method=method,group=group,style=style,metric=metric,**summary([val(agg,method,s,group,style,metric) for s in range(3)])))
     paired.append(dict(aggregation=agg,group=group,style=style,metric=metric,contrast='AttributeOnly-RandomAttribute',**summary([val(agg,METHODS[0],s,group,style,metric)-val(agg,METHODS[1],s,group,style,metric) for s in range(3)])))
     if style!='original':
      targetchange=val('target','OriginalTarget',None,group,style,metric)-val('target','OriginalTarget',None,group,'original',metric)
      for method in METHODS:
       d=[val(agg,method,s,group,style,metric)-val(agg,method,s,group,'original',metric) for s in range(3)]
       changes.append(dict(aggregation=agg,method=method,group=group,style=style,metric=metric,contrast='paraphrase-original',target_change=targetchange,**summary(d)))
       changes.append(dict(aggregation=agg,method=method,group=group,style=style,metric=metric,contrast='(paraphrase-original) minus target change',target_change=targetchange,**summary([v-targetchange for v in d])))
 out=R/'paraphrases/analysis';out.mkdir(exist_ok=True)
 for name,data in [('metrics_by_seed.csv',results),('mean_sd.csv',summaries),('paired_method_deltas.csv',paired),('within_model_changes.csv',changes),('per_question_changes.csv',question_changes),('question_direction_counts.csv',direction_counts)]:csvout(out/name,data)
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 styles=['original','ordinary','substantial']
 for ai,agg in enumerate(['summed','token_mean']):
  for mi,metric in enumerate(['F-Prob','F-ROUGE']):
   ax=axes[ai,mi]
   for method,color in zip(METHODS,['#167d9a','#bc553e']):
    rs=[next(p for p in summaries if p['aggregation']==agg and p['method']==method and p['group']=='all200' and p['style']==style and p['metric']==metric) for style in styles]
    ax.errorbar(range(3),[p['mean'] for p in rs],yerr=[p['sample_sd'] for p in rs],marker='o',capsize=3,color=color,label=method)
   target=[val('target','OriginalTarget',None,'all200',style,metric) for style in styles]
   ax.plot(range(3),target,'s--',color='#333333',label='Original target (one model)')
   ax.set(xticks=range(3),xticklabels=styles,ylabel=metric+' (lower = less target-answer recovery)',title=agg);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('All 200 original IDs; error bars: sample SD over training seeds, fixed masks')
 fig.savefig(out/'paraphrase_metrics.png',dpi=170);fig.savefig(out/'paraphrase_metrics.pdf');plt.close(fig)
 save(out/'scoring_replay.json',replay)
 result=dict(original_target=[p for p in results if p['aggregation']=='target' and p['group']=='all200'],method_paired_all200=[p for p in paired if p['group']=='all200'],within_model_all200=[p for p in changes if p['group']=='all200'],mean_sd_all200=[p for p in summaries if p['group']=='all200'])
 save(out/'summary.json',result);return result
def main():
 archived=op_analysis(read(R/'operating_points/archived_raw_points.json'),'archived_two_points')
 if '--archived-only' in sys.argv:print(json.dumps(archived));return
 points=[]
 for agg in ['summed','token_mean']:
  for method in METHODS:
   for seed in range(3):
    for step in [30,60,90,120]:
     p=R/'evaluations/reproduction'/agg/method/f'seed{seed}/step{step}/official/official_metrics.json'
     points.append(dict(aggregation=agg,method=method,seed=seed,step=step,trajectory='fresh_reproduction',**read(p)))
 new=op_analysis(points,'reproduction_four_points');para=para_analysis()
 old={(p['aggregation'],p['method'],p['seed'],p['step']):p for p in read(R/'operating_points/archived_raw_points.json')}
 comparison=[]
 for p in points:
  k=tuple(p[x] for x in ['aggregation','method','seed','step'])
  if k in old:
   for metric in ['ES','forget_probability','forget_rouge','utility','privleak']:
    comparison.append(dict(aggregation=k[0],method=k[1],seed=k[2],step=k[3],metric=metric,archived=old[k][metric],reproduction=p[metric],delta=p[metric]-old[k][metric]))
 csvout(R/'operating_points/reproduction_vs_archived.csv',comparison)
 save(R/'RESULTS.json',dict(matching=new,paraphrases=para,archived_matching=archived))
 print(json.dumps(dict(analysis_complete=True,measured_checkpoints=len(points),qa_models=13)))
if __name__=='__main__':main()
