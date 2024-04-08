FAQs
====

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_en/migration_guide/faq.rst
    :alt: View Source on Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  use_third_party_op

MindSpore provides a `FAQ <https://mindspore.cn/docs/en/r2.3.0rc1/faq/installation.html>`_ during using MindSpore. This chapter also collates the solutions to the set of common problems mentioned in the migration documentation.

- Environmental Preparations

  **Q: How do I set up a MindSpore environment?**

  A: MindSpore currently supports running on various devices such as Ascend, GPU, CPU. However, you need to pay attention to choosing the matching hardware platform, operating system, and Python version during the installation process, or else there will be a lot of unpredictable errors. For details, please refer to `Installation guide <https://www.mindspore.cn/install/>`_ .

  For more environmental preparation FAQs, please refer to `Environmental Preparation FAQ Analysis <https://www.mindspore.cn/docs/en/r2.3.0rc1/faq/installation.html>`_ .

- Model Analysis and Preparation

  **Q: How can I see how well MindSpore supports the APIs in the migrated code?**

  A: The API automated scanning tool MindSpore Dev Toolkit can be used (recommended), or we can manually query the API mapping table. For details, please refer to `Analyzing API Compliance <https://www.mindspore.cn/docs/en/r2.3.0rc1/migration_guide/analysis_and_preparation.html#analyzing-api-compliance>`_ .

- Constructing Dataset

  **Q: How do I convert a PyTorch `dataset` to a MindSpore `dataset`?**

  A: The customized dataset logic of MindSpore is similar to that of PyTorch. You need to define a `dataset` class containing `__init__`, `__getitem__`, and `__len__` to read your dataset, instantiate the class into an object (for example, `dataset/dataset_generator`), and transfer the instantiated object to `GeneratorDataset` (on MindSpore) or `DataLoader` (on PyTorch). Then, you are ready to load the customized dataset.
  
  MindSpore provides further `map`->`batch` operations based on `GeneratorDataset`. Users can easily add other customized operations to `map` and start `batch`.
  The customized dataset of MindSpore is loaded as follows:

  .. code-block:: python

      # 1 Data enhancement,shuffle,sampler.
      class Mydata:
          def __init__(self):
              np.random.seed(58)
              self.__data = np.random.sample((5, 2))
              self.__label = np.random.sample((5, 1))
          def __getitem__(self, index):
              return (self.__data[index], self.__label[index])
          def __len__(self):
              return len(self.__data)
      dataset_generator = Mydata()
      dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
      # 2 Customized data enhancement
      dataset = dataset.map(operations=pyFunc, {other_params})
      # 3 batch
      dataset = dataset.batch(batch_size, drop_remainder=True)

  **Q: Why does it report an error when iterating over the data: "The actual amount of data read from generator xx is different from generator.len xx, you should adjust generator.len to make them match" ?**

  A: When defining a randomizable datasets, the result returned by the `__len__` method must be the real dataset size, if it is set to a large size, there will be an out-of-bounds problem when `__getitem__` fetches the value. If the size of the dataset is not defined, you can use an iterable dataset, see `Customize dataset <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/beginner/dataset.html>`_ for details.


  **Q: Why does it report an error when iterating over the data: "Invalid Python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names, the size of column_names is:xx and number of returned NumPy array is:xx" ?**

  A: This is because the number of column names specified in the column_names parameter of GeneratorDataset does not match the number of data output by the source parameter.


  **Q: When using GeneratorDataset or map to load/process data, there may be syntax errors, calculation overflow and other issues that cause data errors, how to troubleshoot and debug?**

  A: Observe the error stack information and locate the error code block from the error stack information, add a print or debugging point near the block of code where the error occurred, to further debugging. For details, please refer to `Data Processing Debugging Method 1 <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/minddata_debug.html#method-1-errors-in-data-processing-execution,-print-logs-or-add-debug-points-to-code-debugging>`_ .


  **Q: How to test the each data processing operator in the map operation if data-enhanced map operation error is reported?**

  A: Map operation can be debugged through the execution of individual operators or through data pipeline debugging mode. For details, please refer to `Data Processing Debugging Method 2 <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/minddata_debug.html#method-2-data-enhanced-map-operation-error,-testing-the-each-data-processing-operator-in-the-map-operation>`_ .


  **Q: While training, we will get very many WARNINGs suggesting that our dataset performance is slow, how should we handle this?**
  
  A: It is possible to iterate through the dataset individually and see the processing time for each piece of data to determine how well the dataset is performing. For details, please refer to `Data Processing Debugging Method 3 <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/minddata_debug.html#method-3-testing-data-processing-performance>`_ .


  **Q: In the process of processing data, if abnormal result values are generated due to computational errors, numerical overflow, etc., resulting in operator computation overflow and weight update anomalies during network training, how should we troubleshoot them?**

  A: Turn off shuffling and fix random seeds to ensure reproductivity, and then use tools such as NumPy to quickly verify the results. For details, please refer to `Data Processing Debugging Method 4 <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/minddata_debug.html#method-4-checking-for-exception-data-in-data-processing>`_ .


  For more common data processing problems, please refer to `Analyzing Common Data Processing Problems <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/minddata_debug.html#analyzing-common-data-processing-problems>`_ , and for differences in data processing during migration, please refer to `Data Pre-Processing Differences Between MindSpore And PyTorch <https://www.mindspore.cn/docs/en/r2.3.0rc1/migration_guide/model_development/dataset.html#comparison-of-data-processing-differences>`_ .

- Gradient Derivation

  **Q: How can I implement the backward computation of an operator?**

  A: MindSpore provides an automated interface for gradient derivation, a feature that shields the user from a great deal of the details and process of derivation. However, if there are some special scenarios where the user needs to manually control the calculation of its backward computation, the user can also define its backward computation through the Cell.bprop interface. For details, please refer to `Customize Cell reverse <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/modules/layer.html#custom-cell-reverse>`_ .

  **Q: How to deal with training instability due to gradient overflow?**

  A: Network overflows are usually manifested as loss Nan/INF, the loss suddenly becomes very large. MindSpore provides `dump data <https://www.mindspore.cn/tutorials/experts/en/r2.3.0rc1/debug/dump.html>`_ to get the information about the overflow operator information. When there is gradient underflow in the network, we can use loss scale to support gradient derivation. For details, please refer to `loss scale <https://www.mindspore.cn/docs/en/r2.3.0rc1/migration_guide/model_development/gradient.html#loss-scale>`_; When the network has gradient explosion, you can consider adding gradient trimming. For details, please refer to `gradient cropping <https://www.mindspore.cn/docs/en/r2.3.0rc1/migration_guide/model_development/gradient.html#gradient-cropping>`_ .

- Debugging and Tuning

  **Q: How do I load a pre-trained PyTorch model for fine-tuning on MindSpore?**

  A: Map parameters of PyTorch and MindSpore one by one. No unified conversion script is provided due to flexible network definitions.

  In general, the parameters names and parameters values are saved in the CheckPoint file. After invoking the loading interface of the corresponding framework and obtaining the parameter names and values, construct the object according to the MindSpore format, and then you can directly invoke the MindSpore interface to save as CheckPoint files in the MindSpore format.

  The main work is to compare the parameter names between different frameworks, so that all parameter names in the network of the two frameworks correspond to each other (a map can be used for mapping). The logic of the following code is transforming the parameter format, excluding the corresponding parameter name.

  .. code-block:: python
 
      import torch
      import mindspore as ms

      def pytorch2mindspore(default_file = 'torch_resnet.pth'):
          # read pth file
          par_dict = torch.load(default_file)['state_dict']
          params_list = []
          for name in par_dict:
              param_dict = {}
              parameter = par_dict[name]
              param_dict['name'] = name
              param_dict['data'] = ms.Tensor(parameter.numpy())
              params_list.append(param_dict)
          ms.save_checkpoint(params_list,  'ms_resnet.ckpt')

  **Q: How can I deal with the problem where a loss does not converge or the accuracy doen not meet the standard?**

  A: Substandard accuracy is generally reflected in the loss not converging, but the accuracy is not as expected, which has many complex reasons, and is more difficult to locate. Here are a few guide links for users to troubleshoot the problem one by one.

  `MindSpore Model Accuracy Tuning Practice (1): Common Accuracy Problems, Causes, and Tuning Approach <https://www.hiascend.com/forum/thread-0215121673876901029-1-1.html>`_.

  `MindSpore Model Accuracy Tuning Practice (2): Accuracy Debugging and Tuning Approach <https://www.hiascend.com/forum/thread-0235121941309178031-1-1.html>`_.

  `MindSpore Model Accuracy Tuning Practice (3): Common Accuracy Problems <https://www.hiascend.com/forum/thread-0235121941523411032-1-1.html>`_.

  For more debugging and tuning FAQs, please refer to `Tuning FAQs and Solutions <https://www.mindspore.cn/docs/en/r2.3.0rc1/migration_guide/debug_and_tune.html#debugging-tools>`_ .

  **Q: During model training, the first step takes a long time, how to optimize it?**

  A: During the model training process, the first step contains the network compilation time. If you want to optimize the performance of the first step, you can analyze whether the model compilation can be optimized. For details, please refer to `Static graph network compilation performance optimization <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/static_graph_expert_programming.html>`_.

  **Q: The non-first step takes a long time during model training, how to optimize it?**

  A: The time-consumption of non-first step during model training includes iteration gap, forward-backward computation and iteration trailing. If we want to optimize the performance of non-first step, we need to obtain the iteration trajectory of the network, and then analyze which part is the performance bottleneck, and optimize the performance recently.
     
  For details, please refer to `Performance Tuning Guide <https://www.mindspore.cn/mindinsight/docs/en/r2.2/performance_tuning_guide.html>`_; and `Performance Debugging Examples <https://www.mindspore.cn/mindinsight/docs/en/r2.2/performance_optimization.html>`_ .

  **Q: When loading benchmark weights for model inference to validate the forward process, there is a warning warning that the weights were not loaded successfully, how to solve it?**

  A: During the load_checkpoint process, if there are weights that are not loaded, MindSpore will give a warning prompt. Generally there are two reasons for loading failure: 1, the weight name is not correct; 2, the weight is missing in the network.

  If the weight names don't match, you need to print MindSpore weight names and the benchmark weight names to see if MindSpore weight names have extra prefixes such as backbone or network, and if so, check whether MindSpore adds auto_prefix=False when initializing `Cell <https://www.mindspore.cn/docs/en/r2.3.0rc1/api_python/nn/mindspore.nn.Cell.html>`_ when initializing _ with auto_prefix=False.

  If the weight name is missing, you need to analyze whether it is reasonable or not. If it is reasonable, you can ignore the alarm prompts, if it is not reasonable, you need to analyze whether the network definition is wrong, and locate and modify it.

  **Q: The migration process is tuned using PyNative, and the process is successful. When I switch to Graph mode, why do I get reported errors?**

  A: The behavior of the model for inference in PyNative mode is no different from normal Python code. However, when switching to Graph mode, MindSpore converts Python source code to Intermediate Representation (IR) by means of source code conversion, and optimizes IR graphs on this basis, and finally executes the optimized graphs on hardware devices.
  
  In this step of operation, currently MindSpore does not yet support the complete Python syntax, so there are some limitations in the writing of the construct function.
  
  For example, PyNative mode can directly determine whether a Tensor value is 0, but switching to Graph mode will report an error that it is not supported.

  .. code-block:: python

      if response == 0:
          return loss
      return loss/response

  In similar cases, the code can be modified to:

  .. code-block:: python

      response_gt = max(response, ms.Tensor(1))
      loss = loss/response_gt
      return loss

  See `Static diagram syntax support <https://www.mindspore.cn/docs/en/r2.3.0rc1/note/static_graph_syntax_support.html>`_ for details.

  **Q: What can I do if the error is reported during training: RuntimeError: "Launch kernel failed, name:Default/... What to do" ?**

  A: This type of error is usually because MindSpore does not support a certain operator, and may require the user to implement the operator themselves. For more details, see `PyTorch and MindSpore API mapping table <https://www.mindspore.cn/docs/en/r2.3.0rc1/note/api_mapping/pytorch_api_mapping.html>`_ .

  **Q: How can I effectively locate the cause of an error reported during PyNative dynamic graph migration?**

  A: If you encounter dynamic graph problems, you can set mindspore.set_context(pynative_synchronize=True) to view the error stack to assist in locating them. For details, please refer to `pynative_synchronize description <https://www.mindspore.cn/docs/en/r2.3.0rc1/api_python/mindspore/mindspore.set_context.html?highlight=pynative_synchronize>`_ .

  **Q: How can I effectively locate the cause of an error reported during Graph mode static graph training?**

  A: There are many reasons for static graph errors, and the general failure will be printed in the log. If you can't intuitively get the error information from the log, you can analyze it by export GLOG_v=1 to specify the log level to get more detailed information about the error.

  Meanwhile, when the compilation of computational graphs reports errors, it will automatically save the file analyze_failed.ir, which can help to analyze the location of the error code. For more details, please refer to `Static Graph Mode Error Analysis <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/error_scenario_analysis.html>`_.

  **Q: Out Of Memory error is reported during Graph mode static graph training, what should I do?**

  A: There are two possible reasons for this error: 1. resources are occupied; 2. not enough video memory.
     
  When resources are occupied, release them via pkill -9 python and retrain.
     
  When there is not enough memory, try lowering the batch_size; analyze the memory to see if there are too many communication operators resulting in low overall memory reuse.
     
  For more details, please refer to `Analysis of the problem of insufficient resources <https://www.mindspore.cn/tutorials/en/r2.3.0rc1/advanced/error_analysis/mindrt_debug.html#insufficient-resources>`_ .

  See `Execution Issues <https://www.mindspore.cn/docs/en/r2.3.0rc1/faq/implement_problem.html>`_ for more tuning FAQs.
