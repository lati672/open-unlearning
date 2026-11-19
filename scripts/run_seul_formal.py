"""Fixed SeUL confirmation: LR=1e-5, exactly two updates, seeds 0/1/2."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from transformers import TrainerCallback,TrainingArguments,set_seed
from omegaconf import OmegaConf
from calibrate_seul import original_model,inputs,probe,SOURCE,SMOKE
from run_seul_smoke import FORMAL,TARGET_REV,TOKENIZER_REV,DATA_REV,ForgetOnly,FiniteCheck,save,sha
from trainer.unlearn.seul import SeUL

CALIBRATION=ROOT/'saves/experiments/tofu_forget05_seul_calibration_seed0_v1'
PROTECTED=Path('/workspace/migration_backups/seul-implementation-before-20261119T205634Z/protected_hashes.json')
SOURCES=['scripts/run_seul_formal.py','scripts/calibrate_seul.py','scripts/run_seul_smoke.py',
    'src/trainer/unlearn/seul.py','src/trainer/masking.py','src/trainer/unlearn/base.py',
    'scripts/evaluate_tofu_checkpoint.py','scripts/run_low_confidence.py',
    'src/data/collators.py','configs/trainer/SeUL.yaml']

def fixed_plan():
    return {'phase':'fixed three-seed SeUL confirmation after seed0 tuning',
        'seeds':[0,1,2],'learning_rate':1e-5,'max_steps':2,'training_examples':200,
        'microbatch':2,'gradient_accumulation_steps':16,'effective_batch':32,
        'optimizer':'paged_adamw_32bit','precision':'bfloat16','schedule':'constant',
        'weight_decay':0.,'warmup_steps':0,'max_grad_norm':1.,'retain_loss':None,
        'threshold':'original-model eligible-label mean','mask_sha256':sha(SMOKE/'seul_masks.jsonl'),
        'target_revision':TARGET_REV,'tokenizer_revision':TOKENIZER_REV,'data_revision':DATA_REV,
        'retain_probe_ids':[i*3799//127 for i in range(128)],'forget_probe_ids':list(range(200)),
        'calibration_directory':str(CALIBRATION),
        'selection':'User fixed LR 1e-5 and step2 after seed0 calibration; no retuning across these seeds.',
        'caveat':'Seed0 participated in tuning; three-seed summary is descriptive, not independent held-out confirmation. Seeds1/2 are new.',
        'seed0_policy':'Fresh two-update training; retain calibration artifact and compare weights/metrics.',
        'evaluation_seed':0,'evaluation':'same full official TOFU protocol and pinned retain-only reference',
        'source_sha256':{p:sha(ROOT/p) for p in SOURCES},
        'source_config_sha256':sha(SOURCE),'calibration_plan_sha256':sha(CALIBRATION/'plan.json'),
        'protocol_manifest_sha256':sha(FORMAL/'protocol_manifest.json'),
        'protected_files':json.loads(PROTECTED.read_text())}

def load_plan(root):
    p=json.loads((root/'plan.json').read_text())
    assert p==fixed_plan(),'Frozen plan, source or inputs changed'
    assert all(sha(Path(k))==v for k,v in p['protected_files'].items()),'Protected comparator changed'
    return p

def init(root):
    p=fixed_plan()
    old=json.loads((CALIBRATION/'plan.json').read_text())
    for name,h in old['source_sha256'].items(): assert sha(ROOT/name)==h
    assert all(sha(Path(k))==v for k,v in p['protected_files'].items())
    assert shutil.disk_usage(ROOT).free>12*1024**3,'Need capacity for three checkpoints'
    assert not root.exists(),'Do not overwrite an existing experiment'
    root.mkdir(parents=True)
    save(root/'plan.json',p)
    shutil.copy2(SMOKE/'seul_masks.jsonl',root/'seul_masks.jsonl')
    shutil.copy2(FORMAL/'protocol_manifest.json',root/'official_protocol_manifest.json')
    save(root/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True)})
    (root/'environment_freeze.txt').write_text('\n'.join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')

class Exposure(TrainerCallback):
    def __init__(self,forget,retain,collator,masks,p):
        self.seen=[];self.steps=[]
        self.forget,self.retain,self.collator,self.masks,self.p=forget,retain,collator,masks,p
    def on_step_end(self,args,state,control,model=None,**kwargs):
        # Match the calibration's diagnostic call at step1; it performs no updates.
        was_training=model.training
        diagnostic=probe(model,self.forget,self.retain,self.collator,self.masks,self.p)
        if was_training:model.train()
        self.steps.append({'step':state.global_step,'epoch':state.epoch,'sample_exposures':len(self.seen),
            'exposure_sha256':hashlib.sha256(json.dumps(self.seen).encode()).hexdigest(),'probe':diagnostic})

def train(root,seed):
    p=load_plan(root);assert seed in p['seeds'] and int(os.environ.get('WORLD_SIZE','1'))==1
    output=root/f'seed{seed}';assert not output.exists();output.mkdir()
    set_seed(seed);model,tokenizer=original_model()
    snapshot,forget,retain,collator=inputs((model,tokenizer))
    masks={x['doc_id']:x for x in map(json.loads,(root/'seul_masks.jsonl').read_text().splitlines())}
    assert sha(root/'seul_masks.jsonl')==p['mask_sha256'] and set(masks)==set(range(200))
    assert sum(sum(x['mask']) for x in masks.values())==1762
    assert sum(sum(x['eligible']) for x in masks.values())==7165
    for i in range(200):
        assert forget[i]['input_ids'].tolist()==masks[i]['input_ids']
        assert forget[i]['labels'].tolist()==masks[i]['labels']
    args=TrainingArguments(output_dir=str(output/'run'),max_steps=2,
        per_device_train_batch_size=2,gradient_accumulation_steps=16,learning_rate=1e-5,
        weight_decay=0.,warmup_steps=0,lr_scheduler_type='constant',bf16=True,optim='paged_adamw_32bit',
        logging_steps=1,report_to=[],save_strategy='no',eval_strategy='no',do_eval=False,
        remove_unused_columns=False,seed=seed,data_seed=seed,dataloader_num_workers=0,max_grad_norm=1.)
    finite=FiniteCheck();exposure=Exposure(forget,retain,collator,masks,p)
    trainer=SeUL(model=model,args=args,train_dataset=ForgetOnly(forget,list(range(200))),
        tokenizer=tokenizer,data_collator=collator,callbacks=[finite,exposure],threshold='mean')
    trainer.selection_audit=copy.deepcopy(masks)
    # Recompute every frozen mask from the original target at both endpoints.
    for i in range(200):
        b=collator([forget[i]])
        trainer.original_selection({k:v.to('cuda:0') for k,v in b.items()},verify_original=True)
    save(output/'training_args.json',args.to_dict())
    save(output/'resolved_model.json',json.loads(SOURCE.read_text())['model'])
    save(output/'before.json',probe(model,forget,retain,collator,masks,p))
    original_loss=trainer.compute_loss
    def tracked(model,inputs,return_outputs=False,**kwargs):
        exposure.seen.extend(inputs['forget']['index'].detach().cpu().tolist())
        result=original_loss(model,inputs,return_outputs=return_outputs,**kwargs)
        assert torch.isfinite(result[0] if return_outputs else result)
        return result
    trainer.compute_loss=tracked
    set_seed(seed);result=trainer.train();trainer.save_state()
    assert trainer.state.global_step==2 and len(finite.gradients)==2 and math.isfinite(result.training_loss)
    assert len(exposure.seen)==64 and len(exposure.steps)==2
    assert all(not q.requires_grad and q.grad is None for q in trainer.selection_model.parameters())
    for i in range(200):
        b=collator([forget[i]])
        trainer.original_selection({k:v.to('cuda:0') for k,v in b.items()},verify_original=True)
    assert all(trainer.selection_audit[i]['mask']==masks[i]['mask'] for i in masks)
    checkpoint=output/'checkpoint-2'
    trainer.save_model(str(checkpoint));tokenizer.save_pretrained(checkpoint)
    weights={q.name:sha(q) for q in checkpoint.glob('*.safetensors')};assert weights
    save(output/'exposure.json',{'indices':exposure.seen,'steps':exposure.steps})
    save(output/'after.json',exposure.steps[-1]['probe'])
    save(output/'training_completed.json',{'seed':seed,'steps':2,'learning_rate':1e-5,
        'training_examples':200,'sample_exposures':len(exposure.seen),'epoch':trainer.state.epoch,
        'exposure_sha256':hashlib.sha256(json.dumps(exposure.seen).encode()).hexdigest(),
        'loss':result.training_loss,'gradient_l2_postclip':finite.gradients,
        'all_200_masks_recomputed_before_after_and_unchanged':True,'original_selector_frozen':True,
        'weights_sha256':weights,'source_sha256':p['source_sha256']})
    print('TRAIN_COMPLETE',seed,flush=True)

def evaluate(root,seed):
    p=load_plan(root)
    from evaluate_tofu_checkpoint import configure_eval,validate_outputs
    from run_low_confidence import replay_metrics
    from evals import get_evaluators
    from model import get_model
    output=root/f'seed{seed}';done=json.loads((output/'training_completed.json').read_text())
    assert done['seed']==seed and done['steps']==2
    checkpoint=output/'checkpoint-2'
    for name,h in done['weights_sha256'].items():assert sha(checkpoint/name)==h
    cfg=OmegaConf.create(json.loads((output/'resolved_model.json').read_text()))
    template=OmegaConf.to_container(cfg.template_args)
    for a in [cfg.model_args,cfg.tokenizer_args]:
        a.pop('revision',None);a.pretrained_model_name_or_path=str(checkpoint);a.local_files_only=True
    cfg.model_args.device_map={'':'cuda:0'}
    set_seed(0);model,tokenizer=get_model(cfg);model.eval()
    protocol=json.loads((FORMAL/'protocol_manifest.json').read_text())['protocol']
    reference=FORMAL/'frozen_inputs/TOFU_EVAL.json'
    assert sha(reference)==protocol['inputs'][protocol['retain_only_model']['evaluation_logs']]
    dest=output/'official';dest.mkdir(exist_ok=True)
    config=configure_eval(protocol,dest,reference)
    provenance={'seed':seed,'evaluation_seed':0,'phase':'fixed three-seed confirmation',
        'checkpoint':str(checkpoint),'weights_sha256':done['weights_sha256'],'mask_sha256':p['mask_sha256'],
        'target_revision':TARGET_REV,'tokenizer_revision':TOKENIZER_REV,'data_revision':DATA_REV,
        'reference_sha256':sha(reference),'eval_config':config,'source_sha256':p['source_sha256']}
    if (dest/'provenance.json').exists():assert json.loads((dest/'provenance.json').read_text())==provenance
    else:save(dest/'provenance.json',provenance)
    set_seed(0)
    if not (dest/'validated.json').exists():
        get_evaluators(OmegaConf.create({'tofu':config}))['tofu'].evaluate(model=model,tokenizer=tokenizer,template_args=template)
    logs=json.loads((dest/'TOFU_EVAL.json').read_text());summary=json.loads((dest/'TOFU_SUMMARY.json').read_text())
    metrics,counts=validate_outputs(config,protocol['official_evaluation'],logs,summary,reference)
    replay=replay_metrics(config,logs,json.loads(reference.read_text()))
    assert all(math.isclose(metrics[k],replay[k],rel_tol=1e-10,abs_tol=1e-14) for k in metrics)
    save(dest/'official_metrics.json',metrics)
    save(dest/'validated.json',{'seed':seed,'metrics':metrics,'recomputed':replay,'sample_counts':counts,
        'learning_rate':1e-5,'max_steps':2,'phase':'fixed three-seed confirmation'})
    print('OFFICIAL_COMPLETE',seed,json.dumps(metrics),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['init','train','evaluate'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--seed',type=int,choices=[0,1,2])
    args=parser.parse_args()
    if args.phase=='init':init(args.root)
    elif args.phase=='train':assert args.seed is not None;train(args.root,args.seed)
    else:assert args.seed is not None;evaluate(args.root,args.seed)

if __name__=='__main__':main()
