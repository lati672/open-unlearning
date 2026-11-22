import os,sys,json,subprocess,time,fcntl,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];PY=sys.executable;WORKER=R/'code/worker.py'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state(phase,**kw):
 v=dict(phase=phase,time_utc=datetime.now(timezone.utc).isoformat(),**kw);save(R/'status.json',v);print(json.dumps(v),flush=True)
def command(args,tag,gpus):
 cmd=[PY,'-u',*map(str,args)];env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus}
 record=dict(command=cmd,cwd='/workspace/open-unlearning',environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HOME','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','OMP_NUM_THREADS']},started_utc=datetime.now(timezone.utc).isoformat(),tag=tag)
 with (R/'execution_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 log=(R/'logs'/f'{tag}.log').open('a');p=subprocess.Popen(cmd,cwd='/workspace/open-unlearning',env=env,stdout=log,stderr=subprocess.STDOUT);log.close();return p
def idle():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
 assert not out.strip(),f'Other GPU processes present; do not preempt: {out}'
def clean_new(run):
 # Only this new trajectory, after all its official and paraphrase evaluations verified.
 assert run.is_relative_to(R/'runs')
 removed=[]
 for step in [30,60,90,120]:
  cp=run/f'checkpoint-{step}'
  for n,h in read(cp/'weight_hashes.json').items():
   p=cp/n
   if p.exists():
    assert not p.is_symlink() and sha(p)==h
    removed.append(dict(relative_path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=h));p.unlink()
 save(run/'NEW_WEIGHTS_CLEANUP_RECEIPT.json',dict(reason='All requested evaluations complete and verified; standing project retention preference; limited remaining disk.',files=removed,original_models_or_results_deleted=False,time_utc=datetime.now(timezone.utc).isoformat()))
def main():
 with (R/'queue.lock').open('a') as h:
  fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  t=time.monotonic();idle();state('target_evaluation')
  if not (R/'evaluations/original_target/questions/completed.json').exists():
   assert command([WORKER,'target'],'target','0').wait()==0,'Target eval failed; inspect log'
  for row in read(R/'manifests/run_manifest.json')['runs']:
   agg,method,seed=row['aggregation'],row['method'],row['seed'];tag=f'{agg}.{method}.seed{seed}';run=R/'runs'/agg/method/f'seed{seed}'
   if (run/'NEW_WEIGHTS_CLEANUP_RECEIPT.json').exists():continue
   idle()
   if not (run/'training_completed.json').exists():
    state('training',aggregation=agg,method=method,seed=seed)
    assert command(['-m','torch.distributed.run','--standalone','--nproc_per_node=2',WORKER,'train',agg,method,seed],tag+'.train','0,1').wait()==0,'Training failed; inspect log before any retry'
   for steps in [(30,60),(90,120)]:
    state('official_evaluation',aggregation=agg,method=method,seed=seed,steps=steps)
    ps=[]
    for gpu,step in enumerate(steps):
     ev=R/'evaluations/reproduction'/agg/method/f'seed{seed}/step{step}'
     done=(ev/'official/validated.json').exists() and (step!=120 or (ev/'questions/completed.json').exists())
     if not done:ps.append((step,command([WORKER,'evaluate',agg,method,seed,step],tag+f'.step{step}.eval',str(gpu))))
    failed=[s for s,p in ps if p.wait()!=0]
    assert not failed,f'Evaluation failure {tag} {failed}'
   for step in [30,60,90,120]:assert (R/'evaluations/reproduction'/agg/method/f'seed{seed}/step{step}/official/validated.json').exists()
   assert (R/'evaluations/reproduction'/agg/method/f'seed{seed}/step120/questions/completed.json').exists()
   clean_new(run);state('run_complete',aggregation=agg,method=method,seed=seed)
  idle();state('all_measurements_complete',elapsed_seconds=time.monotonic()-t,training_runs=12,total_optimizer_updates=1440,official_checkpoints=48,target_evaluated_once=True)
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=repr(e));raise
