MindSpore Insight文档
======================

MindSpore Insight是一款可视化调试调优工具，帮助用户获得更优的模型精度和性能。

通过MindSpore Insight，可以可视化地查看训练过程、优化模型性能。用户还可以通过MindSpore Insight提供的命令行方便地搜索超参，迁移模型。

MindSpore Insight包括以下内容：

- 训练过程可视 (`收集Summary数据、查看训练看板 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/summary_record.html>`_)
- `训练溯源及对比 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/lineage_and_scalars_comparison.html>`_
- `性能调优 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling.html>`_

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindinsight/docs/source_zh_cn/images/mindinsight.png" width="700px" alt="" >

代码仓地址： <https://gitee.com/mindspore/mindinsight>

使用MindSpore Insight可视化训练过程
------------------------------------

1. `收集可视化训练数据 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/summary_record.html>`_

   在训练脚本中使用SummaryCollector记录训练信息，再执行训练。

2. `启动MindSpore Insight可视化训练 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/mindinsight_commands.html#启动服务>`_

   启动MindSpore Insight，并通过 ``--summary-base-dir`` 参数指定summary日志文件目录。

3. `查看训练看板 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/dashboard.html>`_

   在浏览器中打开MindSpore Insight访问地址，点击“训练看板”按钮查看详细信息。

使用MindSpore Insight分析模型性能
------------------------------------

1. `收集模型分析数据 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html#准备训练脚本>`_

   在训练脚本中调用MindSpore Profiler相关接口，再执行训练。

2. `启动MindSpore Insight分析模型 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/mindinsight_commands.html>`_

   启动MindSpore Insight服务，并通过 ``--summary-base-dir`` 参数指定性能数据目录。

3. `分析性能数据 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html#数据准备性能分析>`_

   在浏览器中打开MindSpore Insight访问地址，点击“性能分析”按钮查看并分析训练性能数据。

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