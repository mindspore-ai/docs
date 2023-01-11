MindInsight Documents
=====================

MindInsight is a visualized debugging and optimization tool of MindSpore. It visualizes the training process, model performance optimization, and accuracy debugging. You can also use the command line provided by MindInsight to easily search for hyperparameters and migrate models. MindInsight helps you to easily obtain satisfactory model accuracy and performance.

MindInsight provides the following functions:

- Visualized training process (`Collect Summary Record, View Dashboard <https://www.mindspore.cn/mindinsight/docs/en/r1.9/summary_record.html>`_)
- `Training lineage and comparison <https://www.mindspore.cn/mindinsight/docs/en/r1.9/lineage_and_scalars_comparison.html>`_
- `Performance optimization <https://www.mindspore.cn/mindinsight/docs/en/r1.9/performance_profiling.html>`_
- `Accuracy debugging <https://www.mindspore.cn/mindinsight/docs/en/r1.9/debugger.html>`_
- `Hyperparameter optimization <https://www.mindspore.cn/mindinsight/docs/en/r1.9/hyper_parameters_auto_tuning.html>`_
- `Model migration <https://www.mindspore.cn/mindinsight/docs/en/r1.9/migrate_3rd_scripts_mindconverter.html>`_

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/docs/mindinsight/docs/source_en/images/mindinsight_en.png" width="700px" alt="" >

Using MindInsight to Visualize the Training Process
----------------------------------------------------

1. `Collecting Data for Visualization <https://www.mindspore.cn/mindinsight/docs/en/r1.9/summary_record.html>`_

   Use SummaryCollector to record the training information in the training script and then perform the training.

2. `Starting MindInsight for Visualization <https://www.mindspore.cn/mindinsight/docs/en/r1.9/mindinsight_commands.html#start-the-service>`_

   Start the MindInsight service and set the ``--summary-base-dir`` parameter to specify the directory for storing the summary log file.

3. `Viewing Training Dashboard <https://www.mindspore.cn/mindinsight/docs/en/r1.9/dashboard.html>`_

   Open a browser, enter the MindInsight address in the address box, and click Training Dashboard to view details.

Using MindInsight to Analyze the Model Performance
---------------------------------------------------

1. `Collecting Data for Analysis <https://www.mindspore.cn/mindinsight/docs/en/r1.9/performance_profiling_ascend.html#preparing-the-training-script>`_

   Call MindSpore Profiler APIs in the training script and then perform training.

2. `Starting MindInsight for Analysis <https://www.mindspore.cn/mindinsight/docs/en/r1.9/mindinsight_commands.html>`_

   Start the MindInsight service and set the ``--summary-base-dir`` parameter to specify the directory for storing the performance data.

3. `Analyzing Performance Data <https://www.mindspore.cn/mindinsight/docs/en/r1.9/performance_profiling_ascend.html#training-performance>`_

   Open a browser, enter the MindInsight address in the address box, and click Profiling to view and analyze the training performance data.

Using MindInsight to Debug the Model Accuracy
----------------------------------------------

1. `Starting MindInsight in Debugger Mode <https://www.mindspore.cn/mindinsight/docs/en/r1.9/debugger_online.html#launch-mindinsight-in-debugger-mode>`_

   Configure the ``--enable-debugger True`` ``--debugger-port 50051`` parameter to start MindInsight in debugger mode.

2. `Running the Training Script in Debugger Mode <https://www.mindspore.cn/mindinsight/docs/en/r1.9/debugger_online.html#run-the-training-script-in-debug-mode>`_

   Set the environment variable ``export ENABLE_MS_DEBUGGER`` to True to specify the debugger mode for training. Set the debugging service and port to be connected for training: ``export MS_DEBUGGER_HOST=127.0.0.1`` . ``export MS_DEBUGGER_PORT=50051`` .Run the training script.

3. `Setting and Analyzing Watchpoints in MindInsight <https://www.mindspore.cn/mindinsight/docs/en/r1.9/debugger_online.html#debugger-ui-introduction>`_

   Open a browser, enter the MindInsight address in the address box, click the Debugger tab page, set the watchpoints after the training is connected, and analyze the data such as the computational graphs, tensors, and watchpoint hits to identify the root cause of the accuracy problem.

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
   hyper_parameters_auto_tuning
   migrate_3rd_scripts_mindconverter
   performance_profiling
   debugger
   landscape
   mindinsight_commands

.. toctree::
   :maxdepth: 1
   :caption: Tuning Guide

   accuracy_problem_preliminary_location
   accuracy_optimization
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

`Click to download the API documentation <https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/pdf/r1.9/en/mindinsight.pdf>`_