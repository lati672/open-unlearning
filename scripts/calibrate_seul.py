"""Bounded seed-0 LR/early-stop calibration. No formal multi-seed launcher."""
import argparse
import copy
import csv
import gc
import hashlib
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
from omegaconf import OmegaConf
from transformers import TrainerCallback,TrainingArguments,set_seed
from run_seul_smoke import (FORMAL,TARGET_REV,TOKENIZER_REV,DATA_REV,
    ForgetOnly,FiniteCheck,diagnostics,save,sha)
from model import get_model
from data import get_datasets
from data.collators import DataCollatorForSupervisedDataset
from trainer.unlearn.seul import SeUL

LRS={'lr1e-6':1e-6,'lr3e-6':3e-6,'lr1e-5':1e-5}
STEPS=[1,2,4,8]
SMOKE=ROOT/'saves/experiments/tofu_forget05_seul_paper_smoke_v3'
SOURCE=FORMAL/'collected_results/NPO_adaptive_token_mean_forget05_seed0_steps120/resolved_config.json'

def plan():
    return {'phase':'seed0 pilot calibration, not formal three-seed evidence',
        'seed':0,'learning_rates':LRS,'checkpoints':STEPS,'max_steps':8,
        'training_examples':200,'retain_probe_ids':[i*3799//127 for i in range(128)],
        'forget_probe_ids':list(range(200)),'effective_batch_nominal':32,
        'mask_sha256':sha(SMOKE/'seul_masks.jsonl'),'target_revision':TARGET_REV,
        'tokenizer_revision':TOKENIZER_REV,'data_revision':DATA_REV,
        'optimizer':'paged_adamw_32bit','schedule':'constant','warmup_steps':0,
        'weight_decay':0.,'max_grad_norm':1.,'retain_loss':None,
        'screening':{'primary':'max forget answer-NLL increase with retain NLL increase <=0.1 nat',
            'secondary':'max forget answer-NLL increase with retain NLL increase <=0.3 nat',
            'fallback':'if no feasible point: minimum retain increase among points with forget increase >=0.1 nat, otherwise max forget increase',
            'maximum_official_candidates':2,
            'note':'NLL thresholds are exploratory screening heuristics, not utility guarantees. Official metrics are used for confirmation, not claimed independent validation.'},
        'baseline_official':'fresh original-target evaluation with same official protocol',
        'source_sha256':{p:sha(ROOT/p) for p in ['scripts/calibrate_seul.py','src/trainer/unlearn/seul.py',
            'scripts/run_seul_smoke.py','scripts/evaluate_tofu_checkpoint.py','scripts/run_low_confidence.py']}}

def load_plan(root):
    p=json.loads((root/'plan.json').read_text())
    assert p==plan(),'Source, mask or fixed calibration plan changed'
    return p

def inputs(model):
    snapshot=json.loads(SOURCE.read_text());template=snapshot['model']['template_args']
    forget=get_datasets(OmegaConf.create(snapshot['data']['forget']),tokenizer=model[1],template_args=template)
    retain=get_datasets(OmegaConf.create(snapshot['data']['retain']),tokenizer=model[1],template_args=template)
    assert len(forget)==200 and len(retain)==3800
    for data in [forget,retain]:assert all(DATA_REV in Path(x['filename']).parts for x in data.data.cache_files)
    return snapshot,forget,retain,DataCollatorForSupervisedDataset(model[1],index='index')

def original_model():
    cfg=OmegaConf.create(json.loads(SOURCE.read_text())['model'])
    assert cfg.model_args.revision==TARGET_REV and cfg.tokenizer_args.revision==TOKENIZER_REV
    cfg.model_args.local_files_only=True;cfg.tokenizer_args.local_files_only=True
    cfg.model_args.device_map=None
    result=get_model(copy.deepcopy(cfg));result[0].config.use_cache=False
    assert result[0].config._commit_hash==TARGET_REV
    return result

def probe(model,forget,retain,collator,masks,p):
    return {'forget':diagnostics(model,forget,p['forget_probe_ids'],collator,masks),
            'retain':diagnostics(model,retain,p['retain_probe_ids'],collator)}

class Checkpoints(TrainerCallback):
    def __init__(self,root,label,p,forget,retain,collator,masks):
        self.root,self.label,self.p=root,label,p
        self.forget,self.retain,self.collator,self.masks=forget,retain,collator,masks
        self.trainer=None;self.seen=[];self.exposure={}

    def on_step_end(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step)
        self.exposure[step]=len(self.seen)
        if step not in STEPS:return
        was_training=model.training
        values=probe(model,self.forget,self.retain,self.collator,self.masks,self.p)
        if was_training:model.train()
        row={'label':self.label,'learning_rate':LRS[self.label],'step':step,'seed':0,
             'actual_epoch':state.epoch,'sample_exposures':len(self.seen),
             'exposure_sha256':hashlib.sha256(json.dumps(self.seen).encode()).hexdigest(),**values}
        dest=self.root/self.label/f'checkpoint-{step}'
        if shutil.disk_usage(self.root).free<5*1024**3:raise RuntimeError('Insufficient free disk for checkpoint')
        self.trainer.save_model(str(dest));self.trainer.tokenizer.save_pretrained(dest)
        save(dest/'probe.json',row)
        save(dest/'resolved_model.json',json.loads(SOURCE.read_text())['model'])
        print('CALIBRATION_POINT',json.dumps(row),flush=True)

def train(root,label):
    p=load_plan(root);assert int(os.environ.get('WORLD_SIZE','1'))==1
    output=root/label;assert not output.exists();output.mkdir()
    set_seed(0);model,tokenizer=original_model()
    snapshot,forget,retain,collator=inputs((model,tokenizer))
    masks={x['doc_id']:x for x in map(json.loads,(SMOKE/'seul_masks.jsonl').read_text().splitlines())}
    assert set(masks)==set(range(200))
    for i in range(200):assert forget[i]['input_ids'].tolist()==masks[i]['input_ids']
    args=TrainingArguments(output_dir=str(output/'run'),max_steps=8,
        per_device_train_batch_size=2,gradient_accumulation_steps=16,
        learning_rate=LRS[label],weight_decay=0.,warmup_steps=0,lr_scheduler_type='constant',
        bf16=True,optim='paged_adamw_32bit',logging_steps=1,report_to=[],
        save_strategy='no',eval_strategy='no',do_eval=False,remove_unused_columns=False,
        seed=0,data_seed=0,dataloader_num_workers=0,max_grad_norm=1.)
    finite=FiniteCheck();cp=Checkpoints(root,label,p,forget,retain,collator,masks)
    trainer=SeUL(model=model,args=args,train_dataset=ForgetOnly(forget,list(range(200))),
        tokenizer=tokenizer,data_collator=collator,callbacks=[finite,cp],threshold='mean')
    cp.trainer=trainer
    trainer.selection_audit=copy.deepcopy(masks)
    verify_ids=[20*a+q for a in range(10) for q in [0,7]]
    for i in verify_ids:
        batch=collator([forget[i]])
        trainer.original_selection({k:v.to('cuda:0') for k,v in batch.items()},verify_original=True)
    save(output/'training_args.json',args.to_dict())
    save(output/'before.json',probe(model,forget,retain,collator,masks,p))
    original_loss=trainer.compute_loss
    def tracked_loss(model,inputs,return_outputs=False,**kwargs):
        cp.seen.extend(inputs['forget']['index'].detach().cpu().tolist())
        result=original_loss(model,inputs,return_outputs=return_outputs,**kwargs)
        loss=result[0] if return_outputs else result
        if not torch.isfinite(loss):raise ValueError('Non-finite calibration loss')
        return result
    trainer.compute_loss=tracked_loss
    set_seed(0);result=trainer.train();trainer.save_state()
    assert trainer.state.global_step==8 and len(finite.gradients)==8 and math.isfinite(result.training_loss)
    assert all(not q.requires_grad and q.grad is None for q in trainer.selection_model.parameters())
    for i in verify_ids:
        batch=collator([forget[i]])
        trainer.original_selection({k:v.to('cuda:0') for k,v in batch.items()},verify_original=True)
    assert all(trainer.selection_audit[i]['mask']==masks[i]['mask'] for i in masks)
    save(output/'completed.json',{'seed':0,'steps':8,'sample_exposures':len(cp.seen),
        'exposure_sha256':hashlib.sha256(json.dumps(cp.seen).encode()).hexdigest(),
        'loss':result.training_loss,'gradient_l2_postclip':finite.gradients,
        'mask_unchanged':True,'source_sha256':p['source_sha256']})

def select(root):
    p=load_plan(root);rows=[];fingerprints=[]
    for label in LRS:
        done=json.loads((root/label/'completed.json').read_text());fingerprints.append(done['exposure_sha256'])
        before=json.loads((root/label/'before.json').read_text())
        for step in STEPS:
            x=json.loads((root/label/f'checkpoint-{step}/probe.json').read_text())
            row={'key':label+'-step'+str(step),'label':label,'step':step,'learning_rate':LRS[label],
                 'forget_nll':x['forget']['answer_nll'],'retain_nll':x['retain']['answer_nll'],
                 'forget_delta':x['forget']['answer_nll']-before['forget']['answer_nll'],
                 'retain_delta':x['retain']['answer_nll']-before['retain']['answer_nll'],
                 'sample_exposures':x['sample_exposures'],'exposure_sha256':x['exposure_sha256']}
            rows.append(row)
    assert len(set(fingerprints))==1,'Learning rates did not use identical sample order'
    for step in STEPS:assert len({x['exposure_sha256'] for x in rows if x['step']==step})==1
    selected=[]
    for budget in [.1,.3]:
        feasible=[x for x in rows if x['retain_delta']<=budget and x['forget_delta']>0]
        if feasible:
            choice=max(feasible,key=lambda x:(x['forget_delta'],-x['retain_delta'],-x['step']))
            if choice['key'] not in [q['key'] for q in selected]:selected.append(dict(choice,screen_retain_delta_limit=budget))
    if not selected:
        possible=[x for x in rows if x['forget_delta']>=.1]
        choice=min(possible,key=lambda x:x['retain_delta']) if possible else max(rows,key=lambda x:x['forget_delta'])
        selected=[dict(choice,screen_retain_delta_limit=None,fallback=True)]
    with (root/'screening.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(root/'selected.json',{'candidates':selected,'all_points':rows,'official_eval_limit':2})
    print('SELECTED',json.dumps(selected),flush=True)

def official(root,point):
    p=load_plan(root)
    from evaluate_tofu_checkpoint import configure_eval,validate_outputs
    from run_low_confidence import replay_metrics
    from evals import get_evaluators
    if point=='original':
        model,tokenizer=original_model();model.to('cuda:0')
        template=json.loads(SOURCE.read_text())['model']['template_args'];checkpoint=None
    else:
        selected=json.loads((root/'selected.json').read_text())['candidates']
        entry=next(x for x in selected if x['key']==point)
        checkpoint=root/entry['label']/f"checkpoint-{entry['step']}"
        cfg=OmegaConf.create(json.loads((checkpoint/'resolved_model.json').read_text()))
        template=OmegaConf.to_container(cfg.template_args)
        for a in [cfg.model_args,cfg.tokenizer_args]:
            a.pop('revision',None);a.pretrained_model_name_or_path=str(checkpoint);a.local_files_only=True
        cfg.model_args.device_map={'':'cuda:0'}
        model,tokenizer=get_model(cfg)
    model.eval();output=root/'official'/point;output.mkdir(parents=True,exist_ok=True)
    lock=json.loads((FORMAL/'protocol_manifest.json').read_text());protocol=lock['protocol']
    reference=FORMAL/'frozen_inputs/TOFU_EVAL.json'
    assert sha(reference)==protocol['inputs'][protocol['retain_only_model']['evaluation_logs']]
    config=configure_eval(protocol,output,reference)
    provenance={'point':point,'seed':0,'calibration':True,'target_revision':TARGET_REV,
        'mask_sha256':p['mask_sha256'],'reference_sha256':sha(reference),'eval_config':config,
        'checkpoint':str(checkpoint) if checkpoint else None,
        'weights_sha256':{q.name:sha(q) for q in checkpoint.glob('*.safetensors')} if checkpoint else
            {'original_target':json.loads((SMOKE/'availability.json').read_text())['files'][str(Path(os.environ['HF_HOME'])/'models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots'/TARGET_REV/'model.safetensors')]['sha256']}}
    if (output/'provenance.json').exists():assert json.loads((output/'provenance.json').read_text())==provenance
    else:save(output/'provenance.json',provenance)
    set_seed(0)
    if not (output/'validated.json').exists():
        get_evaluators(OmegaConf.create({'tofu':config}))['tofu'].evaluate(model=model,tokenizer=tokenizer,template_args=template)
    logs=json.loads((output/'TOFU_EVAL.json').read_text());summary=json.loads((output/'TOFU_SUMMARY.json').read_text())
    metrics,counts=validate_outputs(config,protocol['official_evaluation'],logs,summary,reference)
    replay=replay_metrics(config,logs,json.loads(reference.read_text()))
    assert all(math.isclose(metrics[k],replay[k],rel_tol=1e-10,abs_tol=1e-14) for k in metrics)
    save(output/'official_metrics.json',metrics)
    save(output/'validated.json',{'metrics':metrics,'recomputed':replay,'sample_counts':counts,
        'formal_three_seed_started':False,'phase':'calibration, seed0 only'})
    print('OFFICIAL_COMPLETE',point,json.dumps(metrics),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['init','train','select','official'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--lr',choices=list(LRS));parser.add_argument('--point')
    args=parser.parse_args();root=args.root.resolve()
    if args.phase=='init':
        assert not root.exists();root.mkdir(parents=True);save(root/'plan.json',plan());return
    if args.phase=='train':assert args.lr;train(root,args.lr)
    elif args.phase=='select':select(root)
    elif args.phase=='official':assert args.point;official(root,args.point)

if __name__=='__main__':main()
