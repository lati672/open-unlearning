import os, sys, json, hashlib, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone

REPO=Path('/workspace/open-unlearning')
R=REPO/'saves/experiments/tofu_forget05_attribute_operating_paraphrase_v1'
SOURCE=REPO/'saves/experiments/tofu_forget05_adaptive_density_phase_a/source_v4'
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'scripts'),str(REPO/'scripts')]
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert not (R/'protocol.json').exists() and not list((R/'runs').rglob('*.safetensors')), 'Do not overwrite a prepared experiment'
 for folder in ['code','manifests','operating_points','paraphrases','evaluations','logs','runs']:(R/folder).mkdir(parents=True,exist_ok=True)
 shutil.copy2(__file__,R/'code/prepare.py')
 from run_low_confidence import replay_metrics
 from datasets import load_dataset
 inventory=[];points=[];preserve={};runs=[];replay=[]
 roots={'summed':'tofu_forget05_attribute_only','token_mean':'tofu_forget05_attribute_only_token_mean'}
 for agg,exp in roots.items():
  old=REPO/'saves/experiments'/exp;lock=read(old/'protocol_manifest.json')
  for rel,d in lock['inputs_sha256'].items():assert sha(old/rel)==d,rel
  for rel,d in lock['source_sha256'].items():assert sha(SOURCE/rel)==d,rel
  for rel in ['protocol_manifest.json','training_signature.json','v4_evaluation_signature.json','mask_audit.json','formal_results.json']:
   dest=R/'manifests'/agg/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,dest);preserve[str((old/rel).relative_to(REPO))]=sha(old/rel)
  ref=REPO/'saves/experiments/tofu_forget05_multiseed_v4_official/frozen_inputs/TOFU_EVAL.json'
  for method in ['AttributeOnly','RandomAttribute']:
   mp=old/'frozen_inputs'/f'{method}.jsonl'
   dst=R/'manifests/frozen_inputs'/mp.name
   dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.exists(): assert sha(dst)==sha(mp)
   else:shutil.copy2(mp,dst)
   preserve[str(mp.relative_to(REPO))]=sha(mp)
   for seed in range(3):
    run=old/'jobs'/method/'runs'/f'seed{seed}'
    cfg=read(run/'resolved_config.json');args=read(run/'training_args.json')
    assert cfg['trainer']['method_args']['normalization']==agg
    new=R/'runs'/agg/method/f'seed{seed}'
    entry=dict(aggregation=agg,method=method,seed=seed,source=str(run.relative_to(REPO)),output=str(new.relative_to(REPO)),kind='fresh reproduction; not continuation',max_optimizer_updates=120,steps=[30,60,90,120],world_size=2,effective_batch_size=32)
    runs.append(entry)
    save(R/'manifests'/agg/method/f'seed{seed}/resolved_config.json',cfg)
    save(R/'manifests'/agg/method/f'seed{seed}/training_args.json',args)
    states=[]
    for step in [60,120]:
     cp=run/f'checkpoint-{step}';ep=cp/'evals/official'
     weights=[p for p in cp.glob('*') if p.is_file() and p.name.endswith(('.safetensors','.bin')) and p.name!='training_args.bin']
     resume=[p.name for p in cp.glob('*') if p.name.startswith(('optimizer','scheduler','rng_state'))]
     assert not weights and not resume
     provenance=read(ep/'evaluation_provenance.json')
     config=provenance['resolved_eval_config'] if 'resolved_eval_config' in provenance else provenance['context']['resolved_eval_config']
     logs=read(ep/'TOFU_EVAL.json');metrics=read(ep/'official_metrics.json')
     recomputed=replay_metrics(config,logs,read(ref))
     import math
     assert all(math.isclose(metrics[k],recomputed[k],rel_tol=1e-10,abs_tol=1e-14) for k in metrics)
     state=read(cp/'trainer_state.json')
     states.append(dict(step=step,weight_files=[],loadable=False,resume_states=resume,evaluation_available=True))
     point=dict(aggregation=agg,method=method,seed=seed,step=step,trajectory='archived',source=str(ep.relative_to(REPO)),**metrics,privleak_abs=abs(metrics['privleak']))
     points.append(point);replay.append(dict(aggregation=agg,method=method,seed=seed,step=step,passed=True,metrics=metrics,recomputed_metrics=recomputed))
     for name in ['TOFU_EVAL.json','TOFU_SUMMARY.json','official_metrics.json','validated.json','evaluation_provenance.json']:
      preserve[str((ep/name).relative_to(REPO))]=sha(ep/name)
     # Store per-sample replay inputs rather than large duplicate candidate outputs.
     trimmed={k:logs[k] for k in ['forget_Q_A_Prob','forget_Q_A_ROUGE','extraction_strength']}
     save(R/'evaluations/archived'/agg/method/f'seed{seed}/step{step}.json',trimmed)
    inventory.append(dict(aggregation=agg,method=method,seed=seed,checkpoints=states,can_exact_resume=False,training_args_sha256=sha(run/'training_args.json'),config_sha256=sha(run/'resolved_config.json')))
 protocol=read(REPO/'saves/experiments/tofu_forget05_attribute_only/protocol_manifest.json')['protocol']
 ds=load_dataset(protocol['dataset']['id'],'forget05',revision=protocol['dataset']['revision'],split='train')
 assert len(ds)==200 and all(protocol['dataset']['revision'] in Path(x['filename']).parts for x in ds.cache_files)
 questions=[dict(original_id=i,question=x['question']) for i,x in enumerate(ds)]
 (R/'paraphrases/original_questions_only.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in questions))
 dataset_hashes={x['filename']:sha(x['filename']) for x in ds.cache_files}
 save(R/'manifests/datasets.json',dataset_hashes)
 save(R/'manifests/original_preservation.json',preserve)
 save(R/'manifests/inventory.json',inventory)
 save(R/'manifests/archived_metric_replay.json',replay)
 save(R/'operating_points/archived_raw_points.json',points)
 save(R/'manifests/run_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),reason='Only two archived points per trajectory and no trained weights; fixed four-point reproduction for matching and step120 paraphrase evaluation.',maximum_runs=12,maximum_updates_per_run=120,runs=runs,prohibited=['SU (Wan et al.)','SeUL (Wang et al.)','MUSE','hyperparameter search','paper or response edits'],checkpoint_retention='Evaluate official suite at all four new points and original/paraphrase questions at step120 before pruning only new weights if storage requires it. Record hashes and cleanup receipts. Preserve original models and prior results.'))
 save(R/'protocol.json',dict(original_protocol=protocol,source=str(SOURCE.relative_to(REPO)),source_code_commit=read(REPO/'saves/experiments/tofu_forget05_attribute_only/protocol_manifest.json')['code_commit'],workspace_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),training=dict(world_size=2,microbatch_per_gpu=4,accumulation=4,effective_batch=32,max_steps=120,checkpoint_steps=[30,60,90,120],seed=[0,1,2],trajectory_kind='fresh reproduction'),matching=dict(frontier='minimize F-Prob and maximize Utility, at least one strict inequality',interpolation='linear within per-seed common frontier support only',grid_points=201,relative_targets=[.25,.5,.75],nearest_calipers=[.005,.01,.02],primary_caliper=.01,tie_break=['lower step'],secondary_same_bracket=True,no_target_step0=True),paraphrases=dict(count=400,original_questions=200,styles=['ordinary','substantial'],generation_inputs='original question only',freeze_before_model_evaluation=True,metrics=['F-Prob','F-ROUGE'],original_target_evaluations=1),created_utc=datetime.now(timezone.utc).isoformat()))
 save(R/'manifests/source_hashes.json',{str(p.relative_to(SOURCE)):sha(p) for folder in ['src','configs','scripts'] for p in (SOURCE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
 (R/'INVENTORY.md').write_text('# Inventory\n\n12 archived trajectories; 24 official evaluations at steps60/120 replayed from per-example outputs. No trained full/incremental weights, optimizer, scheduler or RNG state remain. Exact continuation and direct paraphrase evaluation of these archived models are impossible. Trainer-state JSON and training_args alone are insufficient.\n\n12 fresh reproductions at fixed steps30/60/90/120 are authorized; never splice their checkpoints with archived curves. Original target cache currently lacks weight files and must be restored at its pinned revision. The original tokenizer and TOFU dataset caches are present.\n\nSee manifests/inventory.json, run_manifest.json and original_preservation.json for individual hashes and availability.\n')
 print(json.dumps(dict(root=str(R),archived_evaluations_replayed=len(replay),new_runs=len(runs),questions=len(questions))))
if __name__=='__main__':main()
