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

