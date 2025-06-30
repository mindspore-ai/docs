# 精度调试

## msprobe精度调试工具

msprobe用于协助开发者定位模型训练中的精度问题，提供了包括精度数据采集和分析、精度预检、溢出检测等一系列精度调试功能。

| 功能名（英文）| 简介 | 适用场景/优势  | 当前版本约束 |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [数据采集 <br>（dump）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md)                        | 采集模型训练过程中的API或Cell层级的前反向输入输出数据，包括层次关系、统计值信息、真实数据和调用栈等。          | 1. 将模型中训练的API或Cell的前反向输入输出数据保存下来分析 <br> 2. 模型出现溢出时，可用于查看哪些API或Cell出现了溢出          | 1. API级数据采集仅对[支持列表](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/support_wrap_ops.yaml)中的API进行采集<br>2. 暂不支持采集inplace类API及其上一节点的反向数据 <br>3. 暂不支持参数及参数梯度的采集      |
| [离线预检 <br>（api_accuracy_checker）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/09.accuracy_checker_MindSpore.md) | 为网络中每个API创建用例，检验其精度，并根据不同比对算法综合判定API在NPU上的精度是否达标，快速找出精度差异API。 | 1. 对模型中所有的API做精度初步排查 <br>2. 精度排查不受模型累计误差影响      | 仅支持MindSpore.mint API     |
| [整网比对 <br>（compare）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/11.accuracy_compare_MindSpore.md)              | NPU精度数据与标杆数据的比对，支持MindSpore框架内和与PyTorch跨框架的比对，助力快速定位精度异常API或Cell。       | 1. MindSpore同框架静态图比对 <br>2. MindSpore同框架动态图比对 <br>3. MindSpore vs PyTorch跨框架动态图比对                 | 部分PyTorch的API关联不到MindSpore，需要手动配置映射关系    |
| [溢出检查 <br>（overflow_checker）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/13.overflow_check_MindSpore.md)       | 检测模型计算过程的输入输出，并在溢出时落盘数据，助力用户快速定位溢出位置。                                     | 1. 当模型出现溢出时，可用于定位最先溢出的API或Cell或kernel <br>2. 相比数据采集，性能更优，磁盘压力更小                        | 1. 除具有与数据采集功能相同的约束外，动态图场景下，不支持 Primitive 和 Jit 类 API 的检测 <br>2. 动态图场景下，仅支持检测API或Cell级别溢出 <br>3. 静态图场景下，仅支持检测kernel级别溢出     |
| [无标杆比对 <br>（free_benchmark）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/16.free_benchmarking_MindSpore.md)    | 不依赖标杆数据，通过对算子输入增加微小扰动，计算扰动后输出与原始输出的相对误差，识别有精度风险算子。           | 1. 无标杆数据场景下的算子精度排查 <br>2. 对个别算子进行升精度修复，验证其对模型loss的影响                                     | 1. 仅支持动态图场景 <br>2. 由于需要拷贝输入进行二次执行，所以在遇到大张量的输入时容易发生显存OOM的问题, 特别是反向比对过程。建议配置该功能下[`list`](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/02.config_introduction.md#16-task-配置为-free_benchmark)参数，减少需比对的API数量<br>3. 比对会延长训练时间，整网比对可能会造成严重的耗时膨胀，建议配置该功能下[`list`](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/02.config_introduction.md#16-task-配置为-free_benchmark)参数，减少需比对的API数量<br>4. 不支持“to cpu”操作，不支持预热功能 |
| [可视化比对 <br>（visualization）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/22.visualization_MindSpore.md)        | 解析dump的精度数据，还原模型图结构，比对各层级精度数据，助力理解模型结构、分析精度问题。                       | 1. 整网精度比对定位可疑算子，通过浏览器展示比对结果，支持快速搜索到可疑算子 <br>2. 支持查看模型层级结果，比对模型层级结构差异 | 1. 由于使用整网dump数据，定位的可疑算子受累计误差影响 <br>2. 当模型规模较大时，比对所需时间较长    |
| [训练状态监控 <br>（monitor）](https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/19.monitor.md)                             | 收集模型训练过程中的激活值、梯度和优化器状态，助力分析计算、通信、优化器各部分异常情况。                       | 1. 通过监控模块级统计量指标，快速定位异常模块位置，如loss出现Nan                                                                  | 1. 仅支持模块级别统计量指标分析 <br>2. 仅支持Megatron、DeepSeed框架 <br>3. 会产生少量耗时和显存膨胀       |

## 框架&套件精度调试功能

除上述工具能力外，为方便开发者使用，MindSpore框架提供了[Dump](https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html)功能，支持采集框架精度数据；

在MindSpore Transformers大模型套件集成了[精度在线监控](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/monitor.html)功能。
