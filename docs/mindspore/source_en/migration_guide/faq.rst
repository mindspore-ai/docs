FAQs
====

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/migration_guide/faq.rst
    :alt: View Source on Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  use_third_party_op

MindSpore provides a `FAQ <https://mindspore.cn/docs/en/r2.3/faq/installation.html>`_ during using MindSpore. This chapter also collates the solutions to the set of common problems mentioned in the migration documentation.

- Constructing Dataset

  **Q: Why does it report an error when iterating over the data: The actual amount of data read from generator xx is different from generator.len xx, you should adjust generator.len to make them match ?**

  A: When defining a randomizable datasets, the result returned by the __len__ method must be the real dataset size, if it is set to a large size, there will be an out-of-bounds problem when getitem fetches the value. If the size of the dataset is not defined, you can use an iterable dataset, see `Customize dataset <https://www.mindspore.cn/tutorials/en/r2.3/beginner/dataset.html>`_ for details.


  **Q: Why does it report an error when iterating over the data: Invalid Python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names, the size of column_names is:xx and number of returned NumPy array is:xx ?**

  A: This is because the number of column names specified in the column_names parameter of GeneratorDataset does not match the number of data output by the source parameter.


  **Q: When using GeneratorDataset or map to load/process data, there may be syntax errors, calculation overflow and other issues that cause data errors, how to troubleshoot and debug?**

  A: Observe the error stack information and locate the error code block from the error stack information, add a print or debugging point near the block of code where the error occurred, to further debugging. For details, please refer to `Data Processing Debugging Method 1 <https://www.mindspore.cn/tutorials/en/r2.3/advanced/error_analysis/minddata_debug.html#method-1-errors-in-data-processing-execution,-print-logs-or-add-debug-points-to-code-debugging>`_ .


  **Q: How to test the each data processing operator in the map operation if data-enhanced map operation error is reported?**

  A: Map operation can be debugged through the execution of individual operators or through data pipeline debugging mode. For details, please refer to `Data Processing Debugging Method 2 <https://www.mindspore.cn/tutorials/en/r2.3/advanced/error_analysis/minddata_debug.html#method-2-data-enhanced-map-operation-error,-testing-the-each-data-processing-operator-in-the-map-operation>`_ .


  **Q: While training, we will get very many WARNINGs suggesting that our dataset performance is slow, how should we handle this?**
  
  A: It is possible to iterate through the dataset individually and see the processing time for each piece of data to determine how well the dataset is performing. For details, please refer to `Data Processing Debugging Method 3 <https://www.mindspore.cn/tutorials/en/r2.3/advanced/error_analysis/minddata_debug.html#method-3-testing-data-processing-performance>`_ .


  **Q: In the process of processing data, if abnormal result values are generated due to computational errors, numerical overflow, etc., resulting in operator computation overflow and weight update anomalies during network training, how should we troubleshoot them?**

  A: Turn off shuffling and fix random seeds to ensure reproductivity, and then use tools such as NumPy to quickly verify the results. For details, please refer to `Data Processing Debugging Method 4 <https://www.mindspore.cn/tutorials/en/r2.3/advanced/error_analysis/minddata_debug.html#method-4-checking-for-exception-data-in-data-processing>`_ .


  For more common data processing problems, please refer to `Analyzing Common Data Processing Problems <https://www.mindspore.cn/tutorials/en/r2.3/advanced/error_analysis/minddata_debug.html#analyzing-common-data-processing-problems>`_ , and for differences in data processing during migration, please refer to `Data Pre-Processing Differences Between MindSpore And PyTorch <https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/dataset.html#comparison-of-data-processing-differences>`_ .

- Network Scripts

  `API Mapping and Handling Strategy of Missing API <https://www.mindspore.cn/docs/en/r2.3/migration_guide/analysis_and_preparation.html#analyzing-api-compliance>`_

  `Dynamic Shape Analysis <https://www.mindspore.cn/docs/en/r2.3/migration_guide/dynamic_shape.html>`_ and `Mitigation Program <https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/model_and_cell.html#dynamic-shape-workarounds>`_

  `Mitigation Program for Sparse Characteristic <https://www.mindspore.cn/docs/en/r2.3/migration_guide/sparsity.html>`_

  `Common Syntax Restrictions and Handling Strategies for Static Graphs <https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/model_and_cell.html#dynamic-and-static-graphs>`_

  `Notes for MindSpore Network Writing <https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/model_development.html#considerations-for-mindspore-network-authoring>`_

  `Using Third-party Operator Libraries Based on Customized Interfaces <https://www.mindspore.cn/docs/en/r2.3/migration_guide/use_third_party_op.html>`_

  `Method for Converting PyTorch Models to MindSpore Models <https://www.mindspore.cn/docs/en/r2.3/migration_guide/sample_code.html#model-validation>`_

- Network Debugging

  `Function Debugging <https://www.mindspore.cn/docs/en/r2.3/migration_guide/debug_and_tune.html#function-debugging>`_

  `Precision Debugging <https://www.mindspore.cn/docs/en/r2.3/migration_guide/debug_and_tune.html#accuracy-debugging>`_

  `Performance Debugging <https://www.mindspore.cn/docs/en/r2.3/migration_guide/debug_and_tune.html#performance-tuning>`_

