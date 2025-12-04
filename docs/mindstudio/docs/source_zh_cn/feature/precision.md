# 精度调试

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindstudio/docs/source_zh_cn/feature/precision.md)

msprobe 是 MindStudio Training Tools 工具链下精度调试部分的工具包。主要包括精度预检、溢出检测和精度比对等功能，目前适配 PyTorch 和 MindSpore 框架。msprobe提供多个子工具，侧重不同的训练场景，可以定位模型训练中的精度问题。

<table width="100%">
    <tr>
        <th width="6%">工具</th>
        <th width="12%">功能</th>
        <th>简介</th>
        <th>适用场景/优势</th>
        <th>当前版本限制</th>
    </tr>
    <tr>
        <td rowspan="9">msprobe</td>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/09.accuracy_checker_MindSpore.md">离线精度预检</a></td>
        <td>为网络中每个API创建用例，检验其精度，并根据不同比对算法综合判定API在NPU上的精度是否达标，快速找出精度差异的API。</td>
        <td>1. 对模型中所有的API做精度初步排查 <br>2. 精度排查不受模型累计误差影响</td>
        <td>仅支持mindspore.mint API</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/31.config_check.md">训前配置检查</a></td>
        <td>用于对比两个环境下可能影响训练精度的配置差异，支持MindSpore和PyTorch两个框架，包括：环境变量、三方库版本、训练超参、权重、数据集、随机操作。</td>
        <td>通过比对两个训练环境下的配置差异，提前识别可能会影响精度差异的配置项。</td>
        <td>在使用 MindSpeed-LLM 进行数据采集时，需要注意动态数据采集中的 <a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/31.config_check.md#动态数据采集">apply_patches</a> 函数需要在 MindSpeed-LLM 框架 pretrain_gpt.py 的 megatron_adaptor 函数导入之后执行。</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md">数据采集</a></td>
        <td>采集模型训练过程中的API或Cell层级的前反向输入输出数据，包括层次关系、统计值信息、真实数据和调用栈等。</td>
        <td>1. 将模型中训练的API或Cell的前反向输入输出数据保存下来分析 <br> 2. 模型出现溢出时，可用于查看哪些API或Cell出现了溢出</td>
        <td>1. API级数据采集仅对<a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/mindspore/dump/hook_cell/support_wrap_ops.yaml">支持列表</a>中的API进行采集<br>2. 暂不支持采集inplace类API及其上一节点的反向数据 <br>3. 暂不支持参数及参数梯度的采集</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/13.overflow_check_MindSpore.md">溢出检查</a></td>
        <td>检测模型计算过程中的输入输出，并在溢出时落盘数据，助力用户快速定位溢出位置。</td>
        <td>1. 当模型出现溢出时，可用于定位最先溢出的API或Cell或kernel <br>2. 相比数据采集，性能更优，磁盘压力更小</td>
        <td>1. 除具有与数据采集功能相同的约束外，动态图场景下，不支持 Primitive 和 Jit 类 API 的检测 <br>2. 动态图场景下，仅支持检测API或Cell级别溢出 <br>3. 静态图场景下，仅支持检测kernel级别溢出</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/19.monitor.md">训练状态监控</a></td>
        <td>收集模型训练过程中的激活值、梯度和优化器状态，助力分析计算、通信、优化器各部分的异常情况。</td>
        <td>通过监控模块级统计量指标，快速定位异常模块位置，如loss出现Nan</td>
        <td>1. 仅支持模块级别统计量指标分析 <br>2. 仅支持Megatron、DeepSeed框架 <br>3. 会产生少量耗时和显存膨胀</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/22.visualization_MindSpore.md">可视化比对</a></td>
        <td>解析Dump的精度数据，还原模型图结构，比对各层级精度数据，助力理解模型结构、分析精度问题。</td>
        <td>1. 整网精度比对定位可疑算子，通过浏览器展示比对结果，支持快速搜索到可疑算子 <br>2. 支持查看模型层级结果，比对模型层级结构差异</td>
        <td>1. 由于使用整网Dump数据，定位的可疑算子受累计误差影响 <br>2. 当模型规模较大时，比对所需时间较长</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/11.accuracy_compare_MindSpore.md">compare精度比对</a></td>
        <td>NPU精度数据与标杆数据的比对，支持MindSpore框架内和与PyTorch跨框架的比对，助力快速定位精度异常的API或Cell。</td>
        <td>1. MindSpore同框架静态图比对 <br>2. MindSpore同框架动态图比对 <br>3. MindSpore vs PyTorch跨框架动态图比对</td>
        <td>部分PyTorch的API关联不到MindSpore，需要手动配置映射关系</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/32.ckpt_compare.md">Checkpoint比对</a></td>
        <td>训练过程中或结束后，比较两个不同的Checkpoint，评估模型相似度。</td>
        <td>在模型训练过程中或结束后，可能保存一些检查点文件 (Checkpoint，简称ckpt) 记录当前模型、优化器等训练状态, 工具支持比较两个不同的ckpt，评估模型相似度。</td>
        <td>当前支持Megatron-LM、MindSpeed (PyTorch/MindTorch) 的ckpt比较。支持TP、PP、EP、VPP模型并行；支持megatron.core、megatron.legacy、TransformerEngine的模型实现。</td>
    </tr>
    <tr>
        <td><a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/16.free_benchmarking_MindSpore.md">无标杆比对</a></td>
        <td>不依赖标杆数据，通过对算子输入增加微小扰动，计算扰动后输出与原始输出的相对误差，识别有精度风险的算子。</td>
        <td>1. 无标杆数据场景下的算子精度排查 <br>2. 对个别算子进行升精度修复，验证其对模型loss的影响</td>
        <td>1. 仅支持动态图场景 <br>2. 由于需要拷贝输入进行二次执行，所以在遇到大张量输入时容易发生显存OOM问题, 特别是反向比对过程。建议配置该功能下<a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/02.config_introduction.md#16-task-配置为-free_benchmark">list</a>参数，减少需比对的API数量<br>3. 比对会延长训练时间，整网比对可能会造成严重的耗时膨胀，建议配置该功能下<a href="https://atomgit.com/Ascend/mstt/blob/br_release_MindStudio_8.2.RC1_TR5_20260923/debug/accuracy_tools/msprobe/docs/02.config_introduction.md#16-task-配置为-free_benchmark">list</a>参数，减少需比对的API数量<br>4. 不支持“to cpu”操作，不支持预热功能</td>
    </tr>
</table>
