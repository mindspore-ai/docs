# 调试调优工具概览与安装说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/mindstudio/docs/source_zh_cn/overview.md)

   <table style="width: 100%;">
      <tr>
         <th style="width: 15%;">类型</th>
         <th style="width: 20%;">名称</th>
         <th style="width: 45%;">简介</th>
         <th style="width: 20%;">安装指南</th>
      </tr>
      <tr>
         <td rowspan="2">精度调试</td>
         <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe">msprobe</a></td>
         <td>提供精度数据采集、精度预检、精度比对和溢出检测等精度调试功能。推荐优先选择msprobe 8.1.1版本</td>
         <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/01.installation.md">安装msprobe</a></td>
      </tr>
      <tr>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/22.visualization_MindSpore.md">Tensorboard</a></td>
        <td>可视化比对工具：为msprobe采集的模型结构和精度数据提供可视化比对功能。仅支持MindSpore>=2.4.0版本。</td>
        <td><a href="https://gitee.com/ascend/mstt/blob/br_release_MindStudio_8.1.RC1_TR5_20260623/debug/accuracy_tools/msprobe/docs/22.visualization_MindSpore.md#1依赖安装">安装tb_graph_ascend</a></td>
      </tr>
      <tr>
         <td rowspan="4">性能调优</td>
         <td><a href="https://www.mindspore.cn/tutorials/zh-CN/r2.7.0/debug/profiler.html">Profiler</a></td>
         <td>MindSpore框架内置的性能数据采集、分析工具。推荐优先选择MindSpore 2.7.0版本。</td>
         <td><a href="https://www.mindspore.cn/install">安装MindSpore</a></td>
      </tr>
      <tr>
         <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze">msprof-anaylze</a></td>
         <td>为采集的性能数据提供统计、分析、专家建议等功能。推荐优先选择msprof-anaylze 2.0.2版本</td>
         <td><a href="https://gitee.com/ascend/mstt/tree/br_release_MindStudio_8.1.RC1_TR5_20260623/profiler/msprof_analyze#安装">安装msprof-anaylze</a></td>
      </tr>
      <tr>
         <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/T&ITools/msleaks/atlas_msleaks_0001.html">msleaks</a></td>
         <td>CANN包提供的命令行工具，用于模型训内存问题定位，提供Step内和Step间的内存异常检测能力，包括Step内内存泄漏分析和Step间内存对比分析。</td>
         <td><a href="https://www.mindspore.cn/install">安装MindSpore</a></td>
      </tr>
      <tr>
         <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html">MindStudio Insight</a></td>
         <td>可视化性能调优工具，提供时间线视图、算子耗时、通信瓶颈分析等功能，辅助快速分析模型性能瓶颈。</td>
         <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0005.html">安装MindStudio Insight</a><br><a href=https://www.mindspore.cn/mindstudio/docs/zh-CN/81RC1/version/mindstudio_insight.html>查询版本配套关系</a></td>
      </tr>
   </table>
