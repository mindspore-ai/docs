MindSpore Insight文档
======================

MindSpore Insight是一款可视化调试调优工具，帮助用户获得更优的模型精度和性能。

通过MindSpore Insight，可以可视化地查看训练过程、优化模型性能、调试精度问题。用户还可以通过MindSpore Insight提供的命令行方便地搜索超参，迁移模型。

MindSpore Insight包括以下内容：

- 训练过程可视 (`收集Summary数据、查看训练看板 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/summary_record.html>`_)
- `训练溯源及对比 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/lineage_and_scalars_comparison.html>`_
- `性能调优 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/performance_profiling.html>`_
- `精度调试 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/debugger.html>`_

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.0/docs/mindinsight/docs/source_zh_cn/images/mindinsight.png" width="700px" alt="" >

代码仓地址： <https://gitee.com/mindspore/mindinsight>

关于MindSpore Insight工具停止发布更新的公告
---------------------------------------------

尊敬的用户朋友们：

我们抱着遗憾的心情向大家宣布，因技术规划调整，MindSpore Insight将逐步停止发布更新。具体安排如下：

- **MindSpore Insight：自MindSpore Insight 2.3版本起停止更新发布。**

在过去的时间里，此应用得到了你们的支持与厚爱，我们对此深表感谢。

接下来，为了确保您的使用体验不受影响，我们提供一下替代方案：

- **MindSpore Insight：系统调优数据可视化已整合至MindStudio Insight**，标量可视化、参数分布图可视化和计算图可视化已整合至MindStudio Insight插件，请参阅 `《MindStudio Insight用户指南》 <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_。

我们的用户支撑团队将持续为您提供服务，如有任何问题或需要帮助，请随时通过论坛联系联系我们。

再次感谢您长久以来对MindSpore Insight的支持与信任，我们将继续致力于为您提供更优质的产品和服务。

.. raw:: html

   <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此致</div>

敬礼！

2025年6月9日

使用MindSpore Insight可视化训练过程
------------------------------------

1. `收集可视化训练数据 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/summary_record.html>`_

   在训练脚本中使用SummaryCollector记录训练信息，再执行训练。

2. `启动MindSpore Insight可视化训练 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/mindinsight_commands.html#启动服务>`_

   启动MindSpore Insight，并通过 ``--summary-base-dir`` 参数指定summary日志文件目录。

3. `查看训练看板 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/dashboard.html>`_

   在浏览器中打开MindSpore Insight访问地址，点击“训练看板”按钮查看详细信息。

使用MindSpore Insight分析模型性能
------------------------------------

1. `收集模型分析数据 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/performance_profiling_ascend.html#准备训练脚本>`_

   在训练脚本中调用MindSpore Profiler相关接口，再执行训练。

2. `启动MindSpore Insight分析模型 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/mindinsight_commands.html>`_

   启动MindSpore Insight服务，并通过 ``--summary-base-dir`` 参数指定性能数据目录。

3. `分析性能数据 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/performance_profiling_ascend.html#数据准备性能分析>`_

   在浏览器中打开MindSpore Insight访问地址，点击“性能分析”按钮查看并分析训练性能数据。

使用MindSpore Insight调试模型精度
----------------------------------

1. `以调试模式启动MindSpore Insight <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/debugger_online.html#以调试模式启动mindspore-insight>`_

   通过配置 ``--enable-debugger True`` ``--debugger-port 50051`` 参数使MindSpore Insight以调试模式启动。

2. `以调试模式运行训练脚本 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/debugger_online.html#以调试模式运行训练脚本>`_

   设置环境变量 ``export ENABLE_MS_DEBUGGER=True`` ，将训练指定为调试模式，并设置训练要连接的调试服务和端口： ``export MS_DEBUGGER_HOST=127.0.0.1；`` ``export MS_DEBUGGER_PORT=50051`` ，然后执行训练脚本。

3. `在MindSpore Insight界面设置监测点并分析 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.3/debugger_online.html#设置监测点>`_

   在浏览器中打开MindSpore Insight访问地址，点击“调试器”页签，等待训练连接后，设置监测点，分析计算图、张量、监测点命中等数据，识别精度问题根因。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   mindinsight_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   summary_record
   dashboard
   lineage_and_scalars_comparison
   performance_profiling
   debugger
   landscape
   mindinsight_commands

.. toctree::
   :maxdepth: 1
   :caption: 调优指南

   accuracy_problem_preliminary_location
   accuracy_optimization
   fixing_randomness
   performance_tuning_guide
   performance_optimization

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindinsight.debugger

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 参考文档

   training_visual_design
   graph_visual_design
   tensor_visual_design
   profiler_design
   faq

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE