"""Follow the bounded measurements with CPU-only analysis and independent checks."""
import os,sys,time,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
while True:
 try:s=json.loads((R/'status.json').read_text())
 except json.JSONDecodeError:time.sleep(1);continue
 if s['phase']=='failed':raise RuntimeError('Measurement queue failed; inspect before retry')
 if s['phase']=='all_measurements_complete':break
 time.sleep(30)
env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
for name in ['analyze.py','verify_matching.py','replay_scores.py','finalize.py']:
 cmd=[sys.executable,'-u',str(R/'code'/name)]
 with (R/'execution_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,cwd='/workspace/open-unlearning',environment={'CUDA_VISIBLE_DEVICES':''},started_utc=datetime.now(timezone.utc).isoformat(),tag='postprocess.'+name))+'\n')
 with (R/'logs'/('postprocess.'+name+'.log')).open('x') as f:
  subprocess.run(cmd,cwd='/workspace/open-unlearning',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'POSTPROCESS_COMPLETE.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),all_four_cpu_stages_passed=True,export_pending_manual_interpretation=True),indent=2))
print('CPU analysis, independent scoring/matching replay and preservation verification complete.')
