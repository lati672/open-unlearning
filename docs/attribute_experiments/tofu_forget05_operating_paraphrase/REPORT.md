# TOFU Forget05：Attribute-only 对照的 operating-point 与改写问答评估

两项实验已执行并核查。所有新增结果来自独立复现轨迹；旧论文结果保留，未与新轨迹拼接。本文不修改论文、表格或 reviewer responses，也没有上传 OpenReview。

## 中文解读与两项实验的边界

**实验1：只有有限、依赖插值的 matched-Utility 支持，不能概括成整体优于 Random。** summed NPO 的三个 seed 在各自共同支持上的 ΔF-Prob 为 −0.039047、−0.034604、−0.038319，三seed描述量均值±sample SD 为 **−0.037324±0.002383**。其中seed0仅一个Utility坐标，seed1/2区间宽度只有约0.000640和0.001208；这三个seed描述量混合了单点效应和极窄区间均值。沿同一匹配线段，ES差值为−0.052497±0.006154，absolute PrivLeak为−8.466667±1.990142，但F-ROUGE更高（+0.040098±0.007750）。这些是评估指标的线性插值，不是新训练并实测的中间模型，也不是梯度或优化强度已经相等。

token-mean NPO 只有seed2的共同Utility单点可比较，ΔF-Prob=−0.065779；seed0/1没有共同frontier支持，不能报告三seed均值或SD。**两种aggregation在matched F-Prob的frontier比较都无共同支持，无法判断相同F-Prob时的Utility优势。** 没有外推、插入原target共同点或额外调参制造重叠。

筛选前的观测近邻检查也没有给出统一优势：主caliper0.01下，summed没有符合原共同范围规则的配对；token-mean的seed0有2个、seed2有1个matched-Utility近邻，F-Prob差值均为负。但seed2唯一一个matched-F-Prob近邻（Attribute step30 vs Random step90）的Utility差为 **−0.056750**，F-Prob残差为0.006741。它是包含被支配早期checkpoint的近邻比较，不是精确相等F-Prob的frontier比较；不能忽略，也不能把它升级成另一种整体优势结论。所有未匹配点和残差完整保留。

**实验2：预定义改写后，相对比较方向保持，但不能据此宣称全面遗忘鲁棒性。** 在全200题主结果中，两种aggregation的原问法、ordinary和substantial三种问法下，Attribute−Random的F-Prob差值在三个seed均为负，F-ROUGE差值均为正。两类改写的三seed配对均值±sample SD为：

| Aggregation | 改写 | ΔF-Prob | ΔF-ROUGE |
|---|---|---:|---:|
| summed | ordinary | −0.027330±0.002266 | +0.018591±0.001502 |
| summed | substantial | −0.027816±0.003012 | +0.007935±0.006563 |
| token-mean | ordinary | −0.042441±0.001957 | +0.024224±0.010085 |
| token-mean | substantial | −0.049457±0.003010 | +0.008421±0.001757 |

也就是说，Attribute保持较低参考答案概率，但文本重叠更高的代价也保持。相对F-Prob差距比原问法缩小，不能写成两项指标都更鲁棒。

同一个模型在改写后，**平均**F-Prob和F-ROUGE在所有已测训练seed中都下降；但这掩盖了单题回升。尤其token-mean Attribute的substantial改写，有 **109–117/200题（54.5%–58.5%）** 的参考答案概率上升，尽管整体均值下降。这里“概率上升”并不等于生成回答已正确恢复，部分变化也很小；逐题计数、median、四分位数和target-adjusted描述量见per_question_recovery_summary.csv。不能把均值下降写成没有恢复风险。

原target本身也明显受改写影响：F-Prob从0.878195降至0.675128/0.663438，F-ROUGE从0.794644降至0.526147/0.500549。因此问法改变了难度，平均分数下降不能单独归因为更彻底的遗忘。两个改写共享原问题ID，本轮没有把400问法视作400个独立事实，也没有问题层面显著性检验。

质量与分组结果同样需要保留：所有400改写均进入全200题主分析；严格AI自查下7条有疑点，对应5个原ID，没有人工审核。冻结前198题敏感性分析，以及明确post-hoc的195题分析均单独报告。两类改写的相对指标方向在这两种质量敏感性中保持；但token-mean原问法的F-ROUGE在seed1由全200题的+0.000615变为195题分析的−0.000475，说明接近零的单seed方向会受题目选择影响。15个空mask题没有从主结果删除；空/非空小组只作探索性描述。

**可支持的结论是：** 在本次冻结mask、固定学习率与120步轨迹、两类固定问法范围内，Attribute的低F-Prob/高F-ROUGE权衡在改写后仍可观察；summed有有限的matched-Utility插值证据。尚不能证明统一frontier优势、相同优化强度、统计显著性、独立Random-draw稳定性、隐私优势或全面攻击鲁棒性。源数据中还有生成虚构奖项等多解任务；固定参考答案概率和ROUGE不是完整的事实真值判定。


## 材料与实际协议

- 原12条轨迹的24个 steps60/120 官方评估从逐样本结果复算一致；训练权重已在此前清理，optimizer/scheduler/RNG 状态不足以精确恢复。因此本轮是 fresh reproduction，不是原轨迹续跑。
- 新增恰好12条轨迹：两种 aggregation × 两个方法 × seeds0/1/2，每条120 optimizer updates，保存30/60/90/120；共1440更新、48个官方评估点。没有新增学习率、density、权重或停止点搜索。
- 原 target revision、tokenizer、TOFU revision、冻结 masks 与原数据顺序保持一致。每组1432/7165 eligible tokens（19.99%），15个空mask保留；Random逐样本等数，mask seed1729未重采样。
- 保持双L40S DDP：每卡microbatch4、accumulation4、有效batch32；BF16、FlashAttention2、paged AdamW32bit、LR1e-5、weight decay0.01、原linear schedule和6步warmup、beta0.1、gamma1/alpha1、retain NLL。实际 TrainingArguments 与各原seed配置逐项核对，路径及local rank除外。
- 冻结源代码直接复用；reference最初与target逐参数相同。原代码通过no_grad前向计算reference，其requires_grad标志并非全部False；本轮确认reference未进入optimizer、没有梯度、参数版本未变化。
- 原target在迁移后缺失的权重已按固定revision恢复并校验上游SHA。新权重在本轨迹所有官方和改写评估完成后按既有保留偏好清理，以控制磁盘；文件hash、大小和清理收据保存。未删除任何旧模型或旧实验。


新增复现对照：24个原steps60/120、每个7项官方指标与原归档逐项精确相等：True。absolute PrivLeak另行复核；完整对照见reproduction_vs_archived_all_metrics.csv及manifests/table4_reproduction_validation.json。旧权重未恢复为原字节，以上是新复现模型的指标一致性。

## 实验1：matched operating points

沿用Appendix C：每seed分别最小化F-Prob、最大化Utility，至少一项严格更优时剔除被支配点；只在两方法frontier共同支持内插值。原始target不用于扩展支持，两种aggregation不混合，旧与新轨迹不拼接。共同支持仅为单点时只计算一次matched-point effect，不复制成201个网格点；只有非零长度区间才使用201点网格均值。sample SD仅来自training seeds，并条件于同一冻结mask，不覆盖独立Random draws。

| Aggregation | Seed | 匹配轴 | Frontier共同支持 |
|---|---:|---|---|
| summed | 0 | Utility | 单点 [0.5525102, 0.5525102] |
| summed | 0 | F-Prob | 无共同支持 |
| summed | 1 | Utility | 区间 [0.5491172, 0.5497575] |
| summed | 1 | F-Prob | 无共同支持 |
| summed | 2 | Utility | 区间 [0.5546365, 0.5558444] |
| summed | 2 | F-Prob | 无共同支持 |
| token_mean | 0 | Utility | 无共同支持 |
| token_mean | 0 | F-Prob | 无共同支持 |
| token_mean | 1 | Utility | 无共同支持 |
| token_mean | 1 | F-Prob | 无共同支持 |
| token_mean | 2 | Utility | 单点 [0.5021264, 0.5021264] |
| token_mean | 2 | F-Prob | 无共同支持 |

| Aggregation | 比较 | seed0 | seed1 | seed2 | 均值 ± sample SD | n |
|---|---|---:|---:|---:|---:|---:|
| summed | matched_utility: ΔF-Prob | -0.039047 | -0.034604 | -0.038319 | -0.037324 ± 0.002383 | 3 |
| summed | matched_fprob: ΔUtility | NA | NA | NA | 无共同支持，无法比较 | 0 |
| token_mean | matched_utility: ΔF-Prob | NA | NA | -0.065779 | -0.065779 ± NA | 1 |
| token_mean | matched_fprob: ΔUtility | NA | NA | NA | 无共同支持，无法比较 | 0 |

最近邻使用筛选前原始点及原始共同范围，有放回，主caliper0.01（另完整报告0.005/0.02），距离并列选较早step；所有未匹配点和原因均保存。

| Aggregation | Seed | 匹配轴 | caliper0.01配对数 | 平均Attribute−Random |
|---|---:|---|---:|---:|
| summed | 0 | Utility | 0 | NA |
| summed | 0 | F-Prob | 0 | NA |
| summed | 1 | Utility | 0 | NA |
| summed | 1 | F-Prob | 0 | NA |
| summed | 2 | Utility | 0 | NA |
| summed | 2 | F-Prob | 0 | NA |
| token_mean | 0 | Utility | 2 | -0.086582 |
| token_mean | 0 | F-Prob | 0 | NA |
| token_mean | 1 | Utility | 0 | NA |
| token_mean | 1 | F-Prob | 0 | NA |
| token_mean | 2 | Utility | 1 | -0.090856 |
| token_mean | 2 | F-Prob | 1 | -0.056750 |

辅助ES、F-ROUGE、absolute PrivLeak沿主要匹配的同一插值线段计算；先对每个run取absolute PrivLeak，再插值。完整结果、区间两端、观测/插值标记与原始轨迹见operating_points/。单点效应不能写成区间平均优势，网格平均不等于pointwise dominance，operating-point匹配也不意味着梯度或优化强度相等。

## 实验2：改写问答

- 200条原问题各生成ordinary/substantial两种改写，共400条。生成时只提供问题；模型/版本/effort与采样参数未暴露，记录unknown。生成工具为当前Codex助手，语义检查为生成者AI自查，未声称人工审核。
- 两个源问题（ID117、131）措辞本身含糊，冻结前已标记。400条全部进入全200题主结果；冻结前396条通过AI等价自查，4条源含义不确定。冻结后更严格的纯文本复核又标记了3条可能范围变化（ID124/ordinary、128/ordinary、151/substantial），因此严格自查下393条无新增疑点、7条有疑点，不等同于人工确认有效。没有修改任何冻结问题或根据模型表现删题。
- 同时报告冻结前指定的198题敏感性分析，以及明确标为post-hoc的195题文本质量敏感性分析；后者按文本疑点排除对应5个原ID，不能代替全200题主结果或被称为预先冻结的分析。15条空mask题始终包含在全200题结果中，并另分组报告。
- 问题、prompt、质量记录与哈希均在任何被测模型运行前冻结。自动检查重复、引用约束和新增完整答案泄漏；token级核查生成输入与空答案构造完全相同，teacher-forcing答案仅在assistant标签区。
- 沿用官方QADataset模板、BF16/FlashAttention2、batch32、greedy max_new_tokens200、目标答案几何平均token概率及ROUGE-L recall。原始问题的新评分与对应新官方评估完全一致；改写每题重新前向/生成，不复用原问题缓存。只报告这两个适用指标，不将其称为官方完整评估或全面攻击保证。

| 模型 | 问法 | F-Prob | F-ROUGE |
|---|---|---:|---:|
| Original target（只评估一次） | original | 0.878195 | 0.794644 |
| Original target（只评估一次） | ordinary | 0.675128 | 0.526147 |
| Original target（只评估一次） | substantial | 0.663438 | 0.500549 |
| summed / AttributeOnly | original | 0.185000 ± 0.001985 | 0.481296 ± 0.010435 |
| summed / AttributeOnly | ordinary | 0.138635 ± 0.000826 | 0.385942 ± 0.003849 |
| summed / AttributeOnly | substantial | 0.157874 ± 0.001440 | 0.382669 ± 0.002329 |
| summed / RandomAttribute | original | 0.216182 ± 0.006285 | 0.431225 ± 0.006464 |
| summed / RandomAttribute | ordinary | 0.165965 ± 0.002903 | 0.367351 ± 0.004201 |
| summed / RandomAttribute | substantial | 0.185690 ± 0.002156 | 0.374734 ± 0.004406 |
| token_mean / AttributeOnly | original | 0.068373 ± 0.002423 | 0.425356 ± 0.009726 |
| token_mean / AttributeOnly | ordinary | 0.040920 ± 0.000918 | 0.360572 ± 0.001218 |
| token_mean / AttributeOnly | substantial | 0.043513 ± 0.000966 | 0.360434 ± 0.004552 |
| token_mean / RandomAttribute | original | 0.135086 ± 0.005296 | 0.401377 ± 0.013182 |
| token_mean / RandomAttribute | ordinary | 0.083361 ± 0.002226 | 0.336348 ± 0.008943 |
| token_mean / RandomAttribute | substantial | 0.092971 ± 0.003446 | 0.352014 ± 0.005863 |

逐seed配对差值（Attribute-only − Random–Attribute）：

| Aggregation | 问法 | 指标 | seed0 | seed1 | seed2 | 均值 ± sample SD |
|---|---|---|---:|---:|---:|---:|
| summed | original | F-Prob | -0.035776 | -0.027090 | -0.030678 | -0.031181 ± 0.004365 |
| summed | ordinary | F-Prob | -0.029857 | -0.025482 | -0.026651 | -0.027330 ± 0.002266 |
| summed | substantial | F-Prob | -0.028518 | -0.024515 | -0.030416 | -0.027816 ± 0.003012 |
| summed | original | F-ROUGE | 0.047024 | 0.060867 | 0.042325 | 0.050072 ± 0.009639 |
| summed | ordinary | F-ROUGE | 0.019699 | 0.016882 | 0.019193 | 0.018591 ± 0.001502 |
| summed | substantial | F-ROUGE | 0.002813 | 0.015333 | 0.005659 | 0.007935 ± 0.006563 |
| token_mean | original | F-Prob | -0.069935 | -0.065826 | -0.064377 | -0.066713 ± 0.002883 |
| token_mean | ordinary | F-Prob | -0.044495 | -0.042231 | -0.040597 | -0.042441 ± 0.001957 |
| token_mean | substantial | F-Prob | -0.052680 | -0.048971 | -0.046720 | -0.049457 ± 0.003010 |
| token_mean | original | F-ROUGE | 0.032081 | 0.000615 | 0.039239 | 0.023978 ± 0.020547 |
| token_mean | ordinary | F-ROUGE | 0.015783 | 0.021498 | 0.035392 | 0.024224 ± 0.010085 |
| token_mean | substantial | F-ROUGE | 0.007021 | 0.007849 | 0.010393 | 0.008421 ± 0.001757 |

同一模型的改写−原问题变化，以及扣除Original target问法变化后的描述性差值，见paraphrases/analysis/within_model_changes.csv。两种改写按原问题ID对齐；pair-average先在每个原ID内平均，不把两种表达当作两个独立事实。本轮不做显著性检验或bootstrap。

## 核查、限制与待核实

- 旧24点逐样本复算通过；新48点官方指标复算、13个模型×3种问法评分重放、全部200个ID覆盖、冻结输入和旧归档哈希复核通过。
- 旧权重不可恢复，因此无法对原论文中实际保存过的step120参数直接进行改写评估。新增训练是相同冻结协议和数据顺序的复现，重现实测与原归档的差值逐项保存于operating_points/reproduction_vs_archived.csv；不声称恢复了被删除权重的字节。
- 改写不是独立人工验证，源数据含有虚构/生成型任务及模糊问法；原target也受问法影响，不能将所有分数差异归因于遗忘鲁棒性。概率与ROUGE的方向可能不同，应逐指标解释。
- 3个seed的SD只反映固定mask下训练seed变化；单一Random draw、问题质量、有限训练点、frontier筛选与插值假设仍限制结论。没有relearning、对抗搜索、SU、SeUL或MUSE新增实验。

## 实际资源

双卡训练计时合计 82.20 分钟，训练device-hours 2.7399；官方评估device-hours 0.8806，问题评估device-hours 0.0916（含重放原问题）。这些阶段计时不含部分加载和启动开销。
NVML采样覆盖内GPU占用小时：[1.8617625944915277, 1.8911141780095109]；峰值显存MiB：[43885.0, 43885.0]。采样从第一条训练中途开始，不补算缺失区间。训练PyTorch allocator峰值 13.33 GiB。完整资源记录见resource_usage.json。
