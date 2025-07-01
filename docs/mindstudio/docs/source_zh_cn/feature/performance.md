# 性能调优

<table>
    <tr>
        <th>工具</th>
        <th>功能</th>
        <th>简介</th>
        <th>适用场景/优势 </th>
    </tr>
    <tr>
        <td rowspan="4">Profiler</td>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/profiler.html#方式一-mindsporeprofilerprofile接口使能">性能数据采集</a></td>
        <td>通用性能数据采集、分析功能。</td>
        <td>模型性能未达到预期，需要对模型性能数据进行采集、分析和调优。</td>
    </tr>
    <tr>
        <td><a href="xxx">轻量化打点数据采集 -- 待补充文档</a></td>
        <td>待补充</td>
        <td>待补充</td>
    </tr>
    <tr>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/profiler.html#方式二-动态profiler使能">动态Profiler</a></td>
        <td>在不中断训练流程的前提下，修改配置文件并完成新配置下的性能数据采集任务。</td>
        <td>常稳训练中发现性能劣化、抖动等，期望在不中断训练情况下进行性能分析。</td>
    </tr>
    <tr>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/profiler.html#方式四-离线解析">离线解析数据</a></td>
        <td>对已采集的数据进行离线解析。</td>
        <td>期望在模型运行过程中仅进行性能数据采集，以节省整体运行时间，或对历史已采集数据进行再次解析。</td>
    </tr>
    <tr>
        <td rowspan="5">MindStudio Insight</td>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0034.html">时间线(Timeline)界面</a></td>
        <td>将模型在host、device上的运行详细情况平铺在时间轴上，直观呈现host侧的API耗时情况以及device侧的task耗时，并将host与device进行关联呈现。</td>
        <td>帮助用户快速识别host瓶颈或device瓶颈，同时提供各种筛选分类、专家建议等功能，支撑用户进行深度调优。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0045.html">算子(Operator)界面</a></td>
        <td>呈现计算算子和通信算子耗时数据。</td>
        <td>帮助开发者快速分析由算子耗时导致的性能瓶颈。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0049.html">概览(Summary)界面</a></td>
        <td>提供通信域识别、划分和耗时拆解、分析功能。支持自动识别通信域和用户自行配置通信域；支持按照通信域对比stage耗时、计算耗时和通信耗时。</td>
        <td>分析同一通信域内的切分是否均匀，是否存在通信慢卡和慢链路问题，帮助开发者快速识别问题。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0052.html">通信(Communication)界面</a></td>
        <td>展示集群中全网链路性能以及所有节点的通信性能。</td>
        <td>通过集群通信与计算重叠时间的分析可以找出集群训练中的慢主机或慢节点。</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0041.html">内存(Memory)界面</a></td>
        <td>提供执行过程中内存信息的可视化呈现。</td>
        <td>查看整体内存趋势，以及通过框选峰值区域快速定位到内存消耗过大的算子。</td>
    </tr>
    <tr>
        <td rowspan="3">msprof-anaylze</td>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze/cluster_analyse/README.md">集群分析工具(cluster_anayze)</a></td>
        <td>训练场景的集群性能数据分析工具，主要对基于通信域的迭代内耗时、通信时间、通信矩阵进行分析。</td>
        <td>适用于定位集群内慢卡、慢节点、慢链路等问题。</td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze/compare_tools/README.md">性能比对工具(compare_tools)</a></td>
        <td>对采集的性能数据进行比对分析。</td>
        <td>比较不同硬件/框架下的性能数据，快速识别性能差异点。</td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze/advisor/README.md">专家建议工具(advisor)</a></td>
        <td>分析训练场景的性能数据并给出专家建议。</td>
        <td>待补充</td>
    </tr>
    <tr>
        <td rowspan="1">msleaks</td>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/msleaks/atlas_msleaks_0001.html">内存泄漏检测</a></td>
        <td>提供Step内和Step间的内存异常检测能力，包括Step内内存泄漏分析和Step间内存对比分析。</td>
        <td>适用于分析内存泄漏异常的场景。</td>
    </tr>
</table>
