# 调试调优工具概览与安装说明

<table>
    <tr>
        <th>类型</th>
        <th>名称</th>
        <th>简介</th>
        <th>安装指南</th>
    </tr>
    <tr>
        <td rowspan="2">精度调试</td>
        <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe">msprobe</a></td>
        <td>提供精度数据采集、精度预检、精度比对和溢出检测等精度调试功能。</td>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/01.installation.md">安装msprobe</a></td>
    </tr>
    <tr>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html">Dump</a></td>
        <td>MindSpore框架内置的精度数据采集工具。</td>
        <td><a href="https://www.mindspore.cn/install">安装MindSpore</a></td>
    </tr>
    <tr>
        <td rowspan="4">性能调优</td>
        <td><a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/profiler.html">Profiler</a></td>
        <td>MindSpore框架内置的性能数据采集、分析工具。</td>
        <td><a href="https://www.mindspore.cn/install">安装MindSpore</a></td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze">msprof-anaylze</a></td>
        <td>为采集的性能数据提供统计、分析、专家建议等功能。</td>
        <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze#安装">安装msprof-anaylze</a></td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/msleaks/atlas_msleaks_0001.html">msleaks</a></td>
        <td>用于模型训内存问题定位，提供Step内和Step间的内存异常检测能力，包括Step内内存泄漏分析和Step间内存对比分析。CANN包提供的命令行工具，需安装MindSpore配套CANN包和CANN包下的Toolkit软件包，并按照教程使能环境变量。</td>
        <td><a href="https://www.mindspore.cn/install">安装CANN包</a><br><a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit">安装Toolkit软件包</a></td>
    </tr>
    <tr>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html">MindStudio Insight</a></td>
        <td>可视化性能调优工具，提供时间线视图、算子耗时、通信瓶颈分析等功能，辅助快速分析模型性能瓶颈。</td>
        <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0005.html">安装MindStudio Insight</a><br><a href=version/mindstudio_insight.md>查询版本配套关系</a></td>
    </tr>
</table>
