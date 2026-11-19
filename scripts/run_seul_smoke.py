"""Bounded seed-0 SeUL pilot; this script cannot launch a formal three-seed run."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader,Dataset
from omegaconf import OmegaConf
from transformers import TrainerCallback,TrainingArguments,AutoModelForCausalLM,set_seed
from data import get_datasets
from data.collators import DataCollatorForSupervisedDataset
from model import get_model
from trainer.masking import eligible_positions
from trainer.unlearn.seul import SeUL,label_logprobs

FORMAL=ROOT/'saves/experiments/tofu_forget05_multiseed_v4_official'
TARGET_REV='88e31200b97e4c0c04ae0d2f0b591f427046d192'
TOKENIZER_REV='9213176726f574b556790deb65791e0c5aa438b6'
DATA_REV='324592d84ae4f482ac7249b9285c2ecdb53e3a68'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

class ForgetOnly(Dataset):
    def __init__(self,source,ids):self.source,self.ids=source,ids
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return {'forget':self.source[self.ids[i]]}

class FiniteCheck(TrainerCallback):
    def __init__(self):self.gradients=[]
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        values=[p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]
        norm=torch.stack(values).sum().sqrt().item()
        if not math.isfinite(norm) or norm==0:raise ValueError('Non-finite or zero gradient')
        self.gradients.append(norm)

@torch.inference_mode()
def diagnostics(model,dataset,ids,collator,masks=None):
    model.eval();nll=selected_nll=other_nll=0.;count=selected_count=other_count=0
    for start in range(0,len(ids),4):
        batch=collator([dataset[i] for i in ids[start:start+4]])
        batch={k:v.to('cuda:0') for k,v in batch.items()}
        logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
        valid=eligible_positions(batch);scores=label_logprobs(logits,batch['labels'])
        nll-=scores[valid].sum().item();count+=valid.sum().item()
        if masks is not None:
            for row,doc in enumerate(batch['index'].tolist()):
                attended=batch['attention_mask'][row].bool();v=valid[row,attended];s=scores[row,attended]
                mask=torch.tensor(masks[doc]['mask'],device=s.device,dtype=torch.bool)
                selected_nll-=s[mask].sum().item();selected_count+=mask.sum().item()
                other_nll-=s[v&~mask].sum().item();other_count+=(v&~mask).sum().item()
    result={'answer_nll':nll/count,'eligible_tokens':count}
    if masks is not None:result.update(selected_nll=selected_nll/max(selected_count,1),
        unselected_nll=other_nll/max(other_count,1),selected_tokens=selected_count)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=4);parser.add_argument('--samples',type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.steps<=8 or not 2<=args.samples<=32:raise ValueError('SMOKE ONLY: max 8 steps and 32 examples')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('SMOKE ONLY: one process, seed 0')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source=FORMAL/'collected_results/NPO_adaptive_token_mean_forget05_seed0_steps120/resolved_config.json'
    snapshot=json.loads(source.read_text());model_cfg=OmegaConf.create(snapshot['model'])
    assert model_cfg.model_args.revision==TARGET_REV and model_cfg.tokenizer_args.revision==TOKENIZER_REV
    model_cfg.model_args.local_files_only=True;model_cfg.tokenizer_args.local_files_only=True
    model_cfg.model_args.device_map=None
    set_seed(0);model,tokenizer=get_model(copy.deepcopy(model_cfg));assert model.config._commit_hash==TARGET_REV
    model.config.use_cache=False
    template=snapshot['model']['template_args']
    forget=get_datasets(OmegaConf.create(snapshot['data']['forget']),tokenizer=tokenizer,template_args=template)
    retain=get_datasets(OmegaConf.create(snapshot['data']['retain']),tokenizer=tokenizer,template_args=template)
    assert len(forget)==200 and len(retain)==3800
    for split in [forget,retain]:
        assert all(DATA_REV in Path(x['filename']).parts for x in split.data.cache_files)
    frozen=[json.loads(x) for x in (FORMAL/'frozen_inputs/masks_TOFU_forget05_base.jsonl').read_text().splitlines()]
    collator=DataCollatorForSupervisedDataset(tokenizer,index='index')
    for i in range(200):assert forget[i]['input_ids'].tolist()==frozen[i]['input_ids']
    # Follow original d=32 effective batch, with explicit AdamW/precision adaptation.
    pilot_ids=([20*a+q for q in (0,7,14) for a in range(10)]+[1,21])[:args.samples]
    ta=TrainingArguments(output_dir=str(args.output/'run'),max_steps=args.steps,
        per_device_train_batch_size=2,gradient_accumulation_steps=math.ceil(args.samples/2),
        learning_rate=5e-5,weight_decay=0.,lr_scheduler_type='constant',warmup_steps=0,
        bf16=True,optim='paged_adamw_32bit',logging_steps=1,report_to=[],
        save_strategy='no',eval_strategy='no',do_eval=False,remove_unused_columns=False,
        seed=0,data_seed=0,dataloader_num_workers=0,max_grad_norm=1.)
    checks=FiniteCheck()
    trainer=SeUL(model=model,args=ta,train_dataset=ForgetOnly(forget,pilot_ids),
        tokenizer=tokenizer,data_collator=collator,callbacks=[checks],threshold='mean')
    for start in range(0,200,4):
        batch=collator([forget[i] for i in range(start,min(start+4,200))])
        batch={k:v.to(trainer.accelerator.device) for k,v in batch.items()}
        trainer.original_selection(batch)
    masks=trainer.selection_audit
    eligible=sum(sum(x['eligible']) for x in masks.values());assert eligible==7165
    with (args.output/'seul_masks.jsonl').open('w') as f:
        for i in range(200):f.write(json.dumps(masks[i],allow_nan=False)+'\n')
    budget={'samples':200,'eligible':eligible,'seed_selected':sum(sum(x['seed_mask']) for x in masks.values()),
            'selected':sum(sum(x['mask']) for x in masks.values()),'empty':sum(not any(x['mask']) for x in masks.values()),
            'span_count':sum(len(x['spans']) for x in masks.values())}
    save(args.output/'mask_audit.json',budget)
    # Pin real HF cache content hashes and all training/evaluation JSON inputs.
    hf=Path(os.environ.get('HF_HOME','/workspace/.hf_home'))
    paths=[hf/'models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots'/TARGET_REV/'model.safetensors',
           hf/'models--meta-llama--Llama-3.2-1B-Instruct/snapshots'/TOKENIZER_REV/'tokenizer.json']
    data_dir=hf/'hub/datasets--locuslab--TOFU/snapshots'/DATA_REV
    data_counts={}
    for name in ['forget05','retain95','holdout05','forget05_perturbed','retain_perturbed','real_authors_perturbed','world_facts_perturbed']:
        p=data_dir/(name+'.json');raw=p.read_text()
        try:records=json.loads(raw)
        except json.JSONDecodeError:records=[json.loads(line) for line in raw.splitlines() if line.strip()]
        assert isinstance(records,list) and all(isinstance(row,dict) for row in records)
        data_counts[name]=len(records);paths.append(p)
    assert data_counts=={'forget05':200,'retain95':3800,'holdout05':200,
        'forget05_perturbed':200,'retain_perturbed':400,'real_authors_perturbed':100,'world_facts_perturbed':117}
    save(args.output/'availability.json',{'target_revision':TARGET_REV,'tokenizer_revision':TOKENIZER_REV,
        'data_revision':DATA_REV,'counts':data_counts,'files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}})
    before={'forget':diagnostics(model,forget,pilot_ids,collator,masks),
            'retain':diagnostics(model,retain,list(range(32)),collator)}
    save(args.output/'before.json',before)
    source_paths=['src/trainer/unlearn/seul.py','configs/trainer/SeUL.yaml','scripts/run_seul_smoke.py','tests/test_seul.py']
    save(args.output/'resolved_config.json',{'method':'SeUL-Paper','threshold':'eligible-token arithmetic mean',
        'threshold_ambiguity':'paper prose says average; displayed alpha equation omits /T',
        'reference':'frozen pre-unlearning target; no pretraining base used','gap_fill':'original selected endpoints distance <=2, eligible positions only',
        'forget_loss':'sum selected log probabilities per example; global nonempty-example mean',
        'retain_loss':None,'seed':0,'samples':pilot_ids,'training_args':ta.to_dict(),
        'model':OmegaConf.to_container(model_cfg),'template':template,'mask_audit':budget,
        'source_sha256':{p:sha(ROOT/p) for p in source_paths},
        'scope':'4-step pilot by default, not formal three-seed TOFU results; original paper benchmarks/stopping rule not replicated'})
    print('MASK_AUDIT',json.dumps(budget),flush=True)
    set_seed(0);result=trainer.train();assert trainer.state.global_step==args.steps and math.isfinite(result.training_loss)
    assert len(checks.gradients)==args.steps
    assert all(not p.requires_grad and p.grad is None for p in trainer.selection_model.parameters())
    for start in range(0,200,4):
        batch=collator([forget[i] for i in range(start,min(start+4,200))])
        trainer.original_selection({k:v.to(trainer.accelerator.device) for k,v in batch.items()},verify_original=True)
    after={'forget':diagnostics(model,forget,pilot_ids,collator,masks),
           'retain':diagnostics(model,retain,list(range(32)),collator)}
    save(args.output/'after.json',after)
    checkpoint=args.output/f'checkpoint-{args.steps}';trainer.save_model(str(checkpoint));trainer.save_state()
    tokenizer.save_pretrained(checkpoint)
    # Reload saved weights and compare numerical diagnostics.
    reloaded=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda:0')
    replay=diagnostics(reloaded,forget,pilot_ids[:4],collator,masks)
    direct=diagnostics(model,forget,pilot_ids[:4],collator,masks)
    assert abs(replay['answer_nll']-direct['answer_nll'])<1e-5
    tokenizer2=type(tokenizer).from_pretrained(checkpoint,local_files_only=True)
    assert tokenizer2.get_vocab()==tokenizer.get_vocab()
    summary={'steps':args.steps,'seed':0,'training_examples':args.samples,'training_loss':result.training_loss,
        'gradient_l2_postclip':checks.gradients,'before':before,'after':after,
        'selected_nll_change':after['forget']['selected_nll']-before['forget']['selected_nll'],
        'original_model_frozen':True,'all_200_masks_unchanged_after_training':True,
        'checkpoint_reload_max_nll_difference':abs(replay['answer_nll']-direct['answer_nll']),
        'official_tofu_evaluation':False,'formal_three_seed_started':False,
        'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    save(args.output/'smoke_completed.json',summary);print('SMOKE_COMPLETED',json.dumps(summary),flush=True)

if __name__=='__main__':main()
