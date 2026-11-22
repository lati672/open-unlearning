import json,re,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for attempt in range(5):
 try:s=json.loads((R/'status.json').read_text());break
 except json.JSONDecodeError:time.sleep(.1)
out=dict(phase=s['phase'],aggregation=s.get('aggregation'),method=s.get('method'),seed=s.get('seed'),finished_training=len(list((R/'runs').rglob('training_completed.json'))),finished_runs=len(list((R/'runs').rglob('NEW_WEIGHTS_CLEANUP_RECEIPT.json'))),official_points=len(list((R/'evaluations/reproduction').rglob('validated.json'))),question_models=len(list((R/'evaluations').rglob('questions/completed.json'))))
if s['phase']=='training':
 p=R/'logs'/f"{s['aggregation']}.{s['method']}.seed{s['seed']}.train.log"
 if p.exists():
  steps=re.findall(r'(\d+)/120',p.read_text());out['current_optimizer_update']=int(steps[-1]) if steps else 0
if s['phase']=='failed':out['error']=s['error']
print(json.dumps(out))
