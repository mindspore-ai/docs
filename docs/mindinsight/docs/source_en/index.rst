MindSpore Insight Documents
=============================

MindSpore Insight is a visualized debugging and optimization tool, which helps users achieve better model precision and performance.

MindSpore Insight visualizes the training process, model performance optimization, and accuracy debugging. You can also use the command line provided by MindSpore Insight to easily search for hyperparameters and migrate models.

MindSpore Insight provides the following functions:

- Visualized training process (`Collect Summary Record, View Dashboard <https://www.mindspore.cn/mindinsight/docs/en/r2.3/summary_record.html>`_)
- `Training lineage and comparison <https://www.mindspore.cn/mindinsight/docs/en/r2.3/lineage_and_scalars_comparison.html>`_
- `Performance optimization <https://www.mindspore.cn/mindinsight/docs/en/r2.3/performance_profiling.html>`_
- `Accuracy debugging <https://www.mindspore.cn/mindinsight/docs/en/r2.3/debugger.html>`_

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.0/docs/mindinsight/docs/source_en/images/mindinsight_en.png" width="700px" alt="" >

Code repository address: <https://gitee.com/mindspore/mindinsight>

Announcement on the Discontinuation of MindSpore Insight Tool Updates
-----------------------------------------------------------------------

Dear users

We regret to inform you that MindSpore Insight will gradually cease releasing updates due to technical planning adjustments. The specific arrangement is as follows:

- **MindSpore Insight: MindSpore Insight will no longer update or release new versions after version 2.3.**

Thank you for your past use of this application.

To maintain a smooth user experience, we provide these options:

- **MindSpore Insightt: System optimization data visualization has been integrated into MindStudio Insight** and scalar visualization, parameter distribution visualization, and computational graphs visualization have been integrated into the MindStudio Insight plugins. For details, see the  `MindStudio Insight User Guide <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_.

Our user support team will keep helping you. If you have questions or need heep, cantact us on the forum.

We sincerely thank you for supporting MindSpore Insight over the years. We will keep working to offer you better products and services.

Sincerely!

June 9, 2025

Using MindSpore Insight to Visualize the Training Process
-----------------------------------------------------------

1. `Collecting Data for Visualization <https://www.mindspore.cn/mindinsight/docs/en/r2.3/summary_record.html>`_

   Use SummaryCollector to record the training information in the training script and then perform the training.

2. `Starting MindSpore Insight for Visualization <https://www.mindspore.cn/mindinsight/docs/en/r2.3/mindinsight_commands.html#starting-the-service>`_

   Start the MindSpore Insight service and set the ``--summary-base-dir`` parameter to specify the directory for storing the summary log file.

3. `Viewing Training Dashboard <https://www.mindspore.cn/mindinsight/docs/en/r2.3/dashboard.html>`_

   Open a browser, enter the MindSpore Insight address in the address box, and click Training Dashboard to view details.

Using MindSpore Insight to Analyze the Model Performance
---------------------------------------------------------

1. `Collecting Data for Analysis <https://www.mindspore.cn/mindinsight/docs/en/r2.3/performance_profiling_ascend.html#preparing-the-training-script>`_

   Call MindSpore Profiler APIs in the training script and then perform training.

2. `Starting MindSpore Insight for Analysis <https://www.mindspore.cn/mindinsight/docs/en/r2.3/mindinsight_commands.html>`_

   Start the MindSpore Insight service and set the ``--summary-base-dir`` parameter to specify the directory for storing the performance data.

3. `Analyzing Performance Data <https://www.mindspore.cn/mindinsight/docs/en/r2.3/performance_profiling_ascend.html#training-performance>`_

   Open a browser, enter the MindSpore Insight address in the address box, and click Profiling to view and analyze the training performance data.

Using MindSpore Insight to Debug the Model Accuracy
------------------------------------------------------

1. `Starting MindSpore Insight in Debugger Mode <https://www.mindspore.cn/mindinsight/docs/en/r2.3/debugger_online.html#launch-mindspore-insight-in-debugger-mode>`_

   Configure the ``--enable-debugger True`` ``--debugger-port 50051`` parameter to start MindSpore Insight in debugger mode.

2. `Running the Training Script in Debugger Mode <https://www.mindspore.cn/mindinsight/docs/en/r2.3/debugger_online.html#run-the-training-script-in-debug-mode>`_

   Set the environment variable ``export ENABLE_MS_DEBUGGER`` to True to specify the debugger mode for training. Set the debugging service and port to be connected for training: ``export MS_DEBUGGER_HOST=127.0.0.1`` . ``export MS_DEBUGGER_PORT=50051`` .Run the training script.

3. `Setting and Analyzing Watchpoints in MindSpore Insight <https://www.mindspore.cn/mindinsight/docs/en/r2.3/debugger_online.html#debugger-ui-introduction>`_

   Open a browser, enter the MindSpore Insight address in the address box, click the Debugger tab page, set the watchpoints after the training is connected, and analyze the data such as the computational graphs, tensors, and watchpoint hits to identify the root cause of the accuracy problem.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   mindinsight_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   summary_record
   dashboard
   lineage_and_scalars_comparison
   performance_profiling
   debugger
   landscape
   mindinsight_commands

.. toctree::
   :maxdepth: 1
   :caption: Tuning Guide

   accuracy_problem_preliminary_location
   accuracy_optimization
   fixing_randomness
   performance_tuning_guide
   performance_optimization

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindinsight.debugger

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: References

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