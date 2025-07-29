# 全流程调试调优工具使用指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindstudio/docs/source_zh_cn/guide/get_start.md)

为方便开发者快速上手使用调试调优工具，[《开发工具快速入门》](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0004.html)介绍了精度调试、性能调优过程中工具常用功能的用法，包含使用msprobe工具进行训练前配置检查、训练状态监控、精度数据采集和比对、精度预检，使用Profiler进行性能数据采集，使用msprof-analyze和MindStudio Insight工具进行性能分析等。

<table>
    <tr>
        <th>类型</th>
        <th>功能</th>
        <th>简介</th>
    </tr>
    <tr>
        <td rowspan="6">模型精度调试</td>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0005.html">训练前配置检查</a></td>
        <td>在训练前或精度比对前，对比两个不同环境下，可能影响训练精度的配置差异。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0006.html">训练状态监控</a></td>
        <td>收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0008.html">精度数据采集</a></td>
        <td>采集模型训练过程中API或Module层级的前反向输入输出数据，支持采集的数据包括Module的层次关系、Module或API的输入输出的真实数据和统计值信息、Module或API的调用栈等等。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0010.html">精度预检</a></td>
        <td>在执行训练前使用，会扫描在昇腾NPU环境下训练模型的API，输出精度情况的诊断和分析，综合判定API在NPU上的精度是否达标，从而找出NPU中存在精度问题的API。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0012.html">compare精度比对</a></td>
        <td>比对功能依赖精度数据采集工具采集的数据，计算模型整网NPU侧和标杆设备（如CPU、GPU、NPU等）的误差指标（如余弦相似度、相对误差小于千分之一的比例、最大值误差等），标记可疑的精度异常API或Module，快速定位精度问题根因。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0013.html">分级可视化构图比对</a></td>
        <td>通过 TensorBoard 直观展示图结构、节点数据、依赖关系等。</td>
    </tr>
    <tr>
        <td rowspan="3">模型性能调优</td>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0017.html">性能数据采集</a></td>
        <td>采集原始性能数据，用于精准定位模型训练或推理过程中的性能瓶颈（如算子耗时、内存占用、设备通信延迟等），帮助开发者优化模型执行效率。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0020.html">msprof-analyze工具分析性能数据</a></td>
        <td>对采集到的性能数据进行统计分析，并给出相关的调优建议。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0021.html">MindStudio Insight工具可视化性能数据</a></td>
        <td>可视化呈现真实软硬件运行数据，多维度分析性能瓶颈点。</td>
    </tr>
</table>