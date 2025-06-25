# 调试调优工具概览与安装说明

<table>
    <tr>
        <th>类型</th>
        <th>名称</th>
        <th>简介</th>
        <th>版本配套与安装</th>
    </tr>
    <tr>
        <td rowspan="3">精度调试</td>
        <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe">msprobe</a></td>
        <td>提供精度数据采集、精度预检、精度比对和溢出检测等精度调试功能。</td>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/01.installation.md">版本配套与安装</a></td>
    </tr>
    <tr>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html">Dump</a></td>
        <td>MindSpore框架内置的精度数据采集工具。</td>
        <td><a href="https://www.mindspore.cn/install">安装MindSpore</a>即可使用</td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/22.visualization_MindSpore.md">Tensorboard</a></td>
        <td>可视化比对工具：为msprobe采集的模型结构和精度数据提供可视化比对功能。</td>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/22.visualization_MindSpore.md#1依赖安装">tb_graph_ascend插件安装指导</a><br>与msprobe配套关系：需与msprobe版本相同<br>与MindSpore配套关系：仅支持MindSpore>=2.4版本</td>
    </tr>
    <tr>
        <td rowspan="4">性能调优</td>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/profiler.html">Profiler</a></td>
        <td>MindSpore框架内置的性能数据采集、分析工具。</td>
        <td><a href="https://www.mindspore.cn/install">安装MindSpore</a>即可使用</td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze">msprof-anaylze</a></td>
        <td>为采集的性能数据提供统计、分析、专家建议等功能。</td>
        <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze#安装">安装指导</a><br>与MindSpore配套关系：仅支持MindSpore>=2.5版本</td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/msleaks/atlas_msleaks_0001.html">msleaks</a></td>
        <td>用于模型训内存问题定位，提供Step内和Step间的内存异常检测能力，包括Step内内存泄漏分析和Step间内存对比分析。</td>
        <td>CANN包提供的命令行工具，<a href="https://www.mindspore.cn/install">安装MindSpore配套CANN包</a>即可使用<br><a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit">注意，需安装CANN包下Toolkit软件包，并按照教程使能环境变量</a></td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html">MindStudio Insight</a></td>
        <td>可视化性能调优工具，提供时间线视图、算子耗时、通信瓶颈分析等功能，辅助快速分析模型性能瓶颈。</td>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0005.html">安装</a><br><a href="https://www.hiascend.com/developer/download/community/result?module=sto+cann&sto=8.0.RC1&cann=8.1.RC1.beta1">社区版资源包获取</a><br><a href=version/mindstudio_insight.md>版本配套</a></td>
    </tr>
</table>
