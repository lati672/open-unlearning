"""Independent audit of reported brackets, supports, frontier points and seed SD."""
import csv,json,math
from pathlib import Path
from statistics import mean,stdev
R=Path(__file__).resolve().parents[1];O=R/'operating_points/reproduction_four_points'
def rows(n):return list(csv.DictReader((O/n).open()))
raw=rows('raw_points.csv');front=rows('frontier_points.csv')
assert len(raw)==48 and {int(x['step']) for x in raw}=={30,60,90,120}
point={(x['aggregation'],x['method'],int(x['seed']),int(x['step'])):x for x in raw}
for agg in ['summed','token_mean']:
 for method in ['AttributeOnly','RandomAttribute']:
  for seed in range(3):
   ps=[p for p in raw if p['aggregation']==agg and p['method']==method and int(p['seed'])==seed]
   expected={(float(p['Utility']),float(p['F-Prob'])) for p in ps if not any(float(q['Utility'])>=float(p['Utility']) and float(q['F-Prob'])<=float(p['F-Prob']) and (float(q['Utility'])>float(p['Utility']) or float(q['F-Prob'])<float(p['F-Prob'])) for q in ps)}
   got={(float(p['Utility']),float(p['F-Prob'])) for p in front if p['aggregation']==agg and p['method']==method and int(p['seed'])==seed}
   assert expected==got,(agg,method,seed)
grid=rows('matched_coordinates.csv');supports=json.loads((O/'common_support.json').read_text())
for x in grid:
 agg,seed,axis=x['aggregation'],int(x['seed']),x['match_axis'];target=float(x['target'])
 support=next(p for p in supports if p['aggregation']==agg and p['seed']==seed and p['analysis']==x['analysis'] and p['kind']=='frontier')['common']
 assert support is not None and support[0]<=target<=support[1]
 for method in ['AttributeOnly','RandomAttribute']:
  a=point[agg,method,seed,int(x[method+'_lower_step'])];b=point[agg,method,seed,int(x[method+'_upper_step'])];w=float(x[method+'_upper_weight'])
  assert 0<=w<=1 and float(a[axis])<=target<=float(b[axis])
  assert math.isclose((1-w)*float(a[axis])+w*float(b[axis]),target,rel_tol=1e-12,abs_tol=1e-14)
  for metric in ['F-Prob','Utility','ES','F-ROUGE','absolute PrivLeak']:
   if metric==axis:continue
   v=(1-w)*float(a[metric])+w*float(b[metric]);assert math.isclose(v,float(x[method+'_'+metric]),rel_tol=1e-12,abs_tol=1e-14)
 for key in [k for k in x if k.startswith('delta_')]:
  metric=key.removeprefix('delta_')
  # Union CSV fields are blank on the other matching axis.
  if x[key]=='':continue
  assert math.isclose(float(x[key]),float(x['AttributeOnly_'+metric])-float(x['RandomAttribute_'+metric]),rel_tol=1e-12,abs_tol=1e-14)
averages=rows('matched_effect_by_seed.csv');summary=rows('matched_effect_summary.csv')
for x in averages:
 vals=[float(p['delta_'+x['metric']]) for p in grid if p['aggregation']==x['aggregation'] and p['seed']==x['seed'] and p['analysis']==x['analysis']]
 assert len(vals)==int(x['n_grid_points']) and len(vals) in [1,201] and math.isclose(mean(vals),float(x['value']),rel_tol=1e-12,abs_tol=1e-14)
 assert (x['comparison_type']=='matched_point')==(len(vals)==1)
for x in summary:
 vals=[float(p['value']) for p in averages if p['aggregation']==x['aggregation'] and p['analysis']==x['analysis'] and p['metric']==x['metric']]
 assert len(vals)==int(x['n_seeds']) and math.isclose(mean(vals),float(x['mean']),rel_tol=1e-12,abs_tol=1e-14)
 if len(vals)>1:assert math.isclose(stdev(vals),float(x['sample_sd']),rel_tol=1e-12,abs_tol=1e-14)
result=dict(passed=True,raw_checkpoints=48,frontier_dominance_independently_checked=True,interpolation_rows=len(grid),no_extrapolation=True,secondary_metrics_same_bracket=True,seed_sample_sd_replayed=True,grid_points_not_used_as_independent_n=True)
(O/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
