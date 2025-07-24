MindStudio 文档
=========================================

MindStudio是面向AI开发者提供的全流程工具链，提供了精度、性能、内存调试及可视化能力，帮助开发者高效完成训练开发等任务。

同时，为了方便开发者快速使用，在MindSpore框架内置了精度数据采集、性能数据采集功能；在MindSpore Transformers等大模型套件集成了精度在线监控、性能采集功能。

本文档汇总了MindStudio和MindSpore框架等提供的系列调试工具，并简要介绍这些工具的安装方式、主要功能、入门指导，以及在大模型场景的使用方式。

代码仓地址： <https://gitee.com/ascend/mstt>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能介绍
   :hidden:

   feature/precision
   feature/performance

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 入门指南
   :hidden:

   guide/get_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 大模型调试调优指南
   :hidden:

   guide/large_model

调试调优工具概览与安装说明
--------------------------------

.. raw:: html

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
         <td><a href="https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0005.html">安装MindStudio Insight</a><br><a href=version/mindstudio_insight.html>查询版本配套关系</a></td>
      </tr>
   </table>