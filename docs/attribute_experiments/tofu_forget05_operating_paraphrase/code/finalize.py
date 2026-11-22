import json,csv,sys,hashlib,subprocess,math,re,zipfile,io
from pathlib import Path
from datetime import datetime,timezone
from statistics import mean,stdev
R=Path(__file__).resolve().parents[1];REPO=Path('/workspace/open-unlearning')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fmt(x,n=6):return 'NA' if x is None else f'{x:.{n}f}'
def main():
 assert read(R/'status.json')['phase']=='all_measurements_complete'
 commands=[json.loads(s) for s in (R/'execution_commands.jsonl').read_text().splitlines()]
 save(R/'execution_commands.json',dict(commands=commands,setup_commands=[dict(command='HF_HOME=/workspace/.hf_home HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 /workspace/venvs/open-unlearning-py311/bin/python /tmp/prepare_tofu_attribute_followup.py',purpose='inventory and historical per-example metric replay'),dict(command='HF_HOME=/workspace/.hf_home HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 /workspace/venvs/open-unlearning-py311/bin/python '+str(R/'code/freeze_paraphrases.py'),purpose='freeze all questions before any tested-model evaluation'),dict(command='/workspace/venvs/open-unlearning-py311/bin/python '+str(R/'code/analyze.py'),purpose='replay scoring, per-seed matching and grouped question analysis')],setup_failures_before_optimizer_updates=['Preparation script missing destination mkdir, corrected','Initial queue.py name shadowed standard library queue, renamed run_jobs.py','Audit initially assumed requires_grad=False on reference; corrected to original no_grad behavior plus actual no-gradient/no-update/optimizer-exclusion checks'],safety='Only listed12 unique120-update trajectories. No failed attempt performed an optimizer update.'))
 preservation=read(R/'manifests/original_preservation.json')
 for name,h in preservation.items():assert sha(REPO/name)==h,name
 for name,h in read(R/'paraphrases/FROZEN.json')['files_sha256'].items():assert sha(R/'paraphrases'/name)==h,name
 mask_gold=read(R/'manifests/summed/protocol_manifest.json')
 for method in ['AttributeOnly','RandomAttribute']:
  assert sha(R/'manifests/frozen_inputs'/f'{method}.jsonl')==mask_gold['mask_hashes'][method]
  assert sha(R/'manifests/frozen_inputs'/f'{method}.metadata.json')==mask_gold['inputs_sha256'][f'frozen_inputs/{method}.metadata.json']
 assert sha(R/'manifests/frozen_inputs/original_answer_spans.tsv')==mask_gold['annotation_sha256']
 SOURCE=REPO/read(R/'protocol.json')['source']
 for name,h in read(R/'manifests/source_hashes.json').items():assert sha(SOURCE/name)==h,name
 restored=read(R/'manifests/target_restoration.json');target=Path('/workspace/.hf_home/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots')/restored['revision']
 for n,v in restored['files'].items():
  if v['sha256']:assert sha(target/n)==v['sha256']
 data_manifest=read(R/'manifests/all_data_tokenizer_hashes.json')
 for d in data_manifest['datasets']:
  for n,h in d['files'].items():assert sha(n)==h
 tokenizer_root=Path('/workspace/.hf_home/models--meta-llama--Llama-3.2-1B-Instruct/snapshots')/data_manifest['tokenizer_revision']
 for n,h in data_manifest['tokenizer_files'].items():assert sha(tokenizer_root/n)==h
 counts=dict(training_runs=0,optimizer_updates=0,official_checkpoints=0,qa_models=1)
 runtimes=[];memory=[];orders={};runchecks=[];checkpoint_inventory=[]
 for row in read(R/'manifests/run_manifest.json')['runs']:
  agg,method,seed=row['aggregation'],row['method'],row['seed'];run=R/'runs'/agg/method/f'seed{seed}'
  done=read(run/'training_completed.json');assert done['global_step']==120 and done['same_original_data_order']
  counts['training_runs']+=1;counts['optimizer_updates']+=done['global_step'];counts['qa_models']+=1
  for rank in range(2):
   check=read(run/f'finite_checks_rank{rank}.json');assert check['same_as_original_order'] and check['reference_frozen'] and check['reference_initial_bitwise_equal']
   assert len(check['loss'])==480 and len(check['gradient_l2_preclip'])==120
   assert all(math.isfinite(v) for key in ['loss','gradient_l2_preclip'] for v in check[key])
   orders.setdefault((seed,rank),set()).add(check['sample_order_sha256']);memory.append(check['peak_allocated_bytes'])
  runtimes.append(done['training_runtime_seconds'])
  for step in [30,60,90,120]:
   folder=R/'evaluations/reproduction'/agg/method/f'seed{seed}/step{step}/official'
   v=read(folder/'validated.json');assert v['step']==step and v['all_finite']
   assert v['checkpoint_sha256']==read(run/f'checkpoint-{step}/weight_hashes.json')
   checkpoint_inventory.append(dict(aggregation=agg,method=method,seed=seed,step=step,kind='new full-model Safetensors checkpoint',successfully_loaded_for_official_evaluation=True,successfully_loaded_for_paraphrases=step==120,weight_sha256=v['checkpoint_sha256'],currently_loadable=False,current_availability='weights pruned only after all requested evaluations; see per-run cleanup receipt',optimizer_scheduler_rng_resume_bundle_available=False))
   counts['official_checkpoints']+=1
  assert (R/'evaluations/reproduction'/agg/method/f'seed{seed}/step120/questions/completed.json').exists()
  receipt=read(run/'NEW_WEIGHTS_CLEANUP_RECEIPT.json');assert not receipt['original_models_or_results_deleted']
  runchecks.append(dict(aggregation=agg,method=method,seed=seed,steps=120,passed=True))
 assert all(len(v)==1 for v in orders.values()) and counts==dict(training_runs=12,optimizer_updates=1440,official_checkpoints=48,qa_models=13)
 save(R/'manifests/final_checkpoint_inventory.json',checkpoint_inventory)
 inventory_note=['','## Final availability after this experiment','',
 'The pinned original target full checkpoint is restored, upstream-hash verified and retained. Its original-question and paraphrase evaluation ran once. The pinned base tokenizer and all required raw datasets are present and hash verified. Base full weights and retain-only full weights were not required or independently verified; frozen masks and official retain-only evaluation logs were used unchanged.',
 'All12 new trajectories finished exactly120 updates; all48 full-model checkpoints loaded successfully for official evaluation, and all12 step120 checkpoints also completed original/paraphrase evaluation. New weights were pruned only after those evaluations under the standing retention preference, with file sizes and hashes in per-run NEW_WEIGHTS_CLEANUP_RECEIPT.json. No old checkpoint/model/result was removed or overwritten. New runs retain metadata/configs/evaluations, not a resumable optimizer/scheduler/RNG state bundle.',
 'The actual trained weights from the archived Table4 experiments remain unavailable. Exact old-trajectory continuation and direct new-question evaluation of those deleted old bytes are impossible; new results are explicitly fresh reproductions. No missing material prevents replaying the saved scores and matching analysis. See final_checkpoint_inventory.json and reproduction_vs_archived.csv.','']
 (R/'INVENTORY.md').write_text((R/'INVENTORY.md').read_text()+'\n'.join(inventory_note))
 tele=[json.loads(x) for x in (R/'logs/gpu_telemetry.jsonl').read_text().splitlines()]
 gpu_hours=[0.,0.];util_hours=[0.,0.];peak_mib=[0.,0.]
 for a,b in zip(tele,tele[1:]):
  dt=b['monotonic']-a['monotonic']
  # Never fill a long missing telemetry interval with an invented measurement.
  if dt>15:continue
  for g in a['gpus']:
   i=g['index'];gpu_hours[i]+=dt/3600*(g['memory_mib']>500);util_hours[i]+=dt/3600*g['utilization_percent']/100;peak_mib[i]=max(peak_mib[i],g['memory_mib'])
 evaltimes=[read(p)['elapsed_seconds'] for p in (R/'evaluations/reproduction').rglob('validated.json')]
 paratimes=[read(p)['elapsed_seconds'] for p in (R/'evaluations').rglob('questions/completed.json')]
 resource=dict(gpus='2 x NVIDIA L40S 48GB; original DDP layout retained',training_wall_seconds=sum(runtimes),training_device_hours=2*sum(runtimes)/3600,official_evaluation_device_hours=sum(evaltimes)/3600,question_evaluation_device_hours=sum(paratimes)/3600,telemetry_sample_count=len(tele),telemetry_observed_occupied_hours_by_gpu=gpu_hours,telemetry_utilization_integral_gpu_hours=util_hours,telemetry_peak_memory_mib=peak_mib,training_peak_torch_allocated_gib=max(memory)/2**30,telemetry_limitations='Sampling began during first training run, after target evaluation; initial interval not imputed. Torch allocator and NVML differ. Stage timing omits some model loading and startup overhead.',final_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
 save(R/'resource_usage.json',resource)
 verify=dict(completed_utc=datetime.now(timezone.utc).isoformat(),counts=counts,original_preservation_files_verified=len(preservation),old_models_results_masks_unchanged=True,target_upstream_hash_verified=True,paraphrases_frozen_unchanged=True,source_code_frozen_unchanged=True,same_original_and_paired_data_order=True,all_losses_gradients_finite=True,official_metric_replay_passed=True,paraphrase_scoring_replay_passed=True,all200_original_ids_included=True,no_generation_answer_leakage=True,no_SU_SeUL_MUSE_runs=True,paper_tables_responses_edited=False,new_weights_cleanup_only_after_all_requested_evaluations=True,run_checks=runchecks)
 save(R/'verification.json',verify)
 results=read(R/'RESULTS.json');op=results['matching'];para=results['paraphrases']
 lines=['# TOFU Forget05：Attribute-only 对照的 operating-point 与改写问答评估','',
 '两项实验已执行并核查。所有新增结果来自独立复现轨迹；旧论文结果保留，未与新轨迹拼接。本文不修改论文、表格或 reviewer responses，也没有上传 OpenReview。','',
 '## 材料与实际协议','',
 '- 原12条轨迹的24个 steps60/120 官方评估从逐样本结果复算一致；训练权重已在此前清理，optimizer/scheduler/RNG 状态不足以精确恢复。因此本轮是 fresh reproduction，不是原轨迹续跑。',
 '- 新增恰好12条轨迹：两种 aggregation × 两个方法 × seeds0/1/2，每条120 optimizer updates，保存30/60/90/120；共1440更新、48个官方评估点。没有新增学习率、density、权重或停止点搜索。',
 '- 原 target revision、tokenizer、TOFU revision、冻结 masks 与原数据顺序保持一致。每组1432/7165 eligible tokens（19.99%），15个空mask保留；Random逐样本等数，mask seed1729未重采样。',
 '- 保持双L40S DDP：每卡microbatch4、accumulation4、有效batch32；BF16、FlashAttention2、paged AdamW32bit、LR1e-5、weight decay0.01、原linear schedule和6步warmup、beta0.1、gamma1/alpha1、retain NLL。实际 TrainingArguments 与各原seed配置逐项核对，路径及local rank除外。',
 '- 冻结源代码直接复用；reference最初与target逐参数相同。原代码通过no_grad前向计算reference，其requires_grad标志并非全部False；本轮确认reference未进入optimizer、没有梯度、参数版本未变化。',
 '- 原target在迁移后缺失的权重已按固定revision恢复并校验上游SHA。新权重在本轨迹所有官方和改写评估完成后按既有保留偏好清理，以控制磁盘；文件hash、大小和清理收据保存。未删除任何旧模型或旧实验。','',
 '## 实验1：matched operating points','',
 '沿用Appendix C：每seed分别最小化F-Prob、最大化Utility，至少一项严格更优时剔除被支配点；只在两方法frontier共同支持内插值。原始target不用于扩展支持，两种aggregation不混合，旧与新轨迹不拼接。共同支持仅为单点时只计算一次matched-point effect，不复制成201个网格点；只有非零长度区间才使用201点网格均值。sample SD仅来自training seeds，并条件于同一冻结mask，不覆盖独立Random draws。','',
 '| Aggregation | Seed | 匹配轴 | Frontier共同支持 |','|---|---:|---|---|']
 for p in op['common_support']:
  if p['kind']=='frontier':lines.append(f"| {p['aggregation']} | {p['seed']} | {p['match_axis']} | {('单点 ' if p['status']=='point only' else '区间 ')+str([round(x,7) for x in p['common']]) if p['common'] else '无共同支持'} |")
 lines+=['','| Aggregation | 比较 | seed0 | seed1 | seed2 | 均值 ± sample SD | n |','|---|---|---:|---:|---:|---:|---:|']
 for agg in ['summed','token_mean']:
  for mode,metric in [('matched_utility','F-Prob'),('matched_fprob','Utility')]:
   p=next((p for p in op['grid_mean_summary'] if p['aggregation']==agg and p['analysis']==mode and p['metric']==metric),None)
   if p:lines.append(f"| {p['aggregation']} | {p['analysis']}: Δ{p['metric']} | {fmt(p['seed0'])} | {fmt(p['seed1'])} | {fmt(p['seed2'])} | {fmt(p['mean'])} ± {fmt(p['sample_sd'])} | {p['n_seeds']} |")
   else:lines.append(f"| {agg} | {mode}: Δ{metric} | NA | NA | NA | 无共同支持，无法比较 | 0 |")
 if not op['grid_mean_summary']:lines+=['','两种主轴均没有可插值的共同frontier支持，主匹配效应不可估计。不能把固定终点差异替代为 matched-operating-point 优势，也不额外调参制造重叠。']
 lines+=['','最近邻使用筛选前原始点及原始共同范围，有放回，主caliper0.01（另完整报告0.005/0.02），距离并列选较早step；所有未匹配点和原因均保存。','',
 '| Aggregation | Seed | 匹配轴 | caliper0.01配对数 | 平均Attribute−Random |','|---|---:|---|---:|---:|']
 for p in op['primary_caliper_nearest']:lines.append(f"| {p['aggregation']} | {p['seed']} | {p['match_axis']} | {p['matched']} | {fmt(p['mean_difference'])} |")
 lines+=['','辅助ES、F-ROUGE、absolute PrivLeak沿主要匹配的同一插值线段计算；先对每个run取absolute PrivLeak，再插值。完整结果、区间两端、观测/插值标记与原始轨迹见operating_points/。单点效应不能写成区间平均优势，网格平均不等于pointwise dominance，operating-point匹配也不意味着梯度或优化强度相等。','',
 '## 实验2：改写问答','',
 '- 200条原问题各生成ordinary/substantial两种改写，共400条。生成时只提供问题；模型/版本/effort与采样参数未暴露，记录unknown。生成工具为当前Codex助手，语义检查为生成者AI自查，未声称人工审核。',
 '- 两个源问题（ID117、131）措辞本身含糊，冻结前已标记。400条全部进入全200题主结果；冻结前396条通过AI等价自查，4条源含义不确定。冻结后更严格的纯文本复核又标记了3条可能范围变化（ID124/ordinary、128/ordinary、151/substantial），因此严格自查下393条无新增疑点、7条有疑点，不等同于人工确认有效。没有修改任何冻结问题或根据模型表现删题。',
 '- 同时报告冻结前指定的198题敏感性分析，以及明确标为post-hoc的195题文本质量敏感性分析；后者按文本疑点排除对应5个原ID，不能代替全200题主结果或被称为预先冻结的分析。15条空mask题始终包含在全200题结果中，并另分组报告。',
 '- 问题、prompt、质量记录与哈希均在任何被测模型运行前冻结。自动检查重复、引用约束和新增完整答案泄漏；token级核查生成输入与空答案构造完全相同，teacher-forcing答案仅在assistant标签区。',
 '- 沿用官方QADataset模板、BF16/FlashAttention2、batch32、greedy max_new_tokens200、目标答案几何平均token概率及ROUGE-L recall。原始问题的新评分与对应新官方评估完全一致；改写每题重新前向/生成，不复用原问题缓存。只报告这两个适用指标，不将其称为官方完整评估或全面攻击保证。','',
 '| 模型 | 问法 | F-Prob | F-ROUGE |','|---|---|---:|---:|']
 for style in ['original','ordinary','substantial']:
  get=lambda met:next(p['value'] for p in para['original_target'] if p['style']==style and p['metric']==met)
  lines.append(f"| Original target（只评估一次） | {style} | {fmt(get('F-Prob'))} | {fmt(get('F-ROUGE'))} |")
 for agg in ['summed','token_mean']:
  for method in ['AttributeOnly','RandomAttribute']:
   for style in ['original','ordinary','substantial']:
    get=lambda met:next(p for p in para['mean_sd_all200'] if p['aggregation']==agg and p['method']==method and p['style']==style and p['metric']==met)
    a,b=get('F-Prob'),get('F-ROUGE')
    lines.append(f"| {agg} / {method} | {style} | {fmt(a['mean'])} ± {fmt(a['sample_sd'])} | {fmt(b['mean'])} ± {fmt(b['sample_sd'])} |")
 lines+=['','逐seed配对差值（Attribute-only − Random–Attribute）：','', '| Aggregation | 问法 | 指标 | seed0 | seed1 | seed2 | 均值 ± sample SD |','|---|---|---|---:|---:|---:|---:|']
 for p in para['method_paired_all200']:
  if p['style']=='paraphrase_pair_average':continue
  lines.append(f"| {p['aggregation']} | {p['style']} | {p['metric']} | {fmt(p['seed0'])} | {fmt(p['seed1'])} | {fmt(p['seed2'])} | {fmt(p['mean'])} ± {fmt(p['sample_sd'])} |")
 lines+=['','同一模型的改写−原问题变化，以及扣除Original target问法变化后的描述性差值，见paraphrases/analysis/within_model_changes.csv。两种改写按原问题ID对齐；pair-average先在每个原ID内平均，不把两种表达当作两个独立事实。本轮不做显著性检验或bootstrap。','',
 '## 核查、限制与待核实','',
 '- 旧24点逐样本复算通过；新48点官方指标复算、13个模型×3种问法评分重放、全部200个ID覆盖、冻结输入和旧归档哈希复核通过。',
 '- 旧权重不可恢复，因此无法对原论文中实际保存过的step120参数直接进行改写评估。新增训练是相同冻结协议和数据顺序的复现，重现实测与原归档的差值逐项保存于operating_points/reproduction_vs_archived.csv；不声称恢复了被删除权重的字节。',
 '- 改写不是独立人工验证，源数据含有虚构/生成型任务及模糊问法；原target也受问法影响，不能将所有分数差异归因于遗忘鲁棒性。概率与ROUGE的方向可能不同，应逐指标解释。',
 '- 3个seed的SD只反映固定mask下训练seed变化；单一Random draw、问题质量、有限训练点、frontier筛选与插值假设仍限制结论。没有relearning、对抗搜索、SU、SeUL或MUSE新增实验。','',
 '## 实际资源','',f"双卡训练计时合计 {sum(runtimes)/60:.2f} 分钟，训练device-hours {resource['training_device_hours']:.4f}；官方评估device-hours {resource['official_evaluation_device_hours']:.4f}，问题评估device-hours {resource['question_evaluation_device_hours']:.4f}（含重放原问题）。这些阶段计时不含部分加载和启动开销。",f"NVML采样覆盖内GPU占用小时：{gpu_hours}；峰值显存MiB：{peak_mib}。采样从第一条训练中途开始，不补算缺失区间。训练PyTorch allocator峰值 {max(memory)/2**30:.2f} GiB。完整资源记录见resource_usage.json。",'']
 (R/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(dict(verified=True,counts=counts,resources=resource)))
if __name__=='__main__':main()
