MindSpore Insight Documents
=============================

MindSpore Insight is a visualized debugging and optimization tool, which helps users achieve better model precision and performance.

MindSpore Insight visualizes the training process, model performance optimization. You can also use the command line provided by MindSpore Insight to easily search for hyperparameters and migrate models.

MindSpore Insight provides the following functions:

- Visualized training process (`Collect Summary Record, View Dashboard <https://www.mindspore.cn/mindinsight/docs/en/master/summary_record.html>`_)
- `Training lineage and comparison <https://www.mindspore.cn/mindinsight/docs/en/master/lineage_and_scalars_comparison.html>`_
- `Performance optimization <https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling.html>`_

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindinsight/docs/source_en/images/mindinsight_en.png" width="700px" alt="" >

Code repository address: <https://gitee.com/mindspore/mindinsight>

Using MindSpore Insight to Visualize the Training Process
-----------------------------------------------------------

1. `Collecting Data for Visualization <https://www.mindspore.cn/mindinsight/docs/en/master/summary_record.html>`_

   Use SummaryCollector to record the training information in the training script and then perform the training.

2. `Starting MindSpore Insight for Visualization <https://www.mindspore.cn/mindinsight/docs/en/master/mindinsight_commands.html#starting-the-service>`_

   Start the MindSpore Insight service and set the ``--summary-base-dir`` parameter to specify the directory for storing the summary log file.

3. `Viewing Training Dashboard <https://www.mindspore.cn/mindinsight/docs/en/master/dashboard.html>`_

   Open a browser, enter the MindSpore Insight address in the address box, and click Training Dashboard to view details.

Using MindSpore Insight to Analyze the Model Performance
---------------------------------------------------------

1. `Collecting Data for Analysis <https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html#preparing-the-training-script>`_

   Call MindSpore Profiler APIs in the training script and then perform training.

2. `Starting MindSpore Insight for Analysis <https://www.mindspore.cn/mindinsight/docs/en/master/mindinsight_commands.html>`_

   Start the MindSpore Insight service and set the ``--summary-base-dir`` parameter to specify the directory for storing the performance data.

3. `Analyzing Performance Data <https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html#training-performance>`_

   Open a browser, enter the MindSpore Insight address in the address box, and click Profiling to view and analyze the training performance data.

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