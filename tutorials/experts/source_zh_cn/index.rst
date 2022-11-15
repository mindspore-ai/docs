.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

深度开发
=====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 数据处理

   dataset/augment
   dataset/cache
   dataset/optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 图编译

   network/control_flow
   network/op_overload
   network/jit_class
   network/constexpr
   network/dependency_control

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型训练优化

   others/execution_opt
   others/gradient_accumulation
   others/adaptive_summation
   others/dimention_reduce_training
   others/thor

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 自定义算子

   operation/op_custom
   operation/ms_kernel
   operation/op_custom_adv

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 自动向量化

   vmap/vmap
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型推理

   infer/inference
   infer/cpu_gpu_mindir
   infer/ascend_910_mindir
   infer/ascend_310_mindir
   infer/ascend_310_air
   infer/model_compression

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 调试调优
   :titlesonly:

   debug/function_debug
   debug/performance_optimization
   精度调优↗ <https://mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_problem_preliminary_location.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行
   :titlesonly:

   parallel/introduction
   parallel/parallel_training_quickstart
   parallel/communicate_ops
   parallel/distributed_case
   parallel/distributed_inference
   parallel/save_load
   parallel/fault_recover
   parallel/multi_dimensional
   parallel/resilience_train_and_predict
   parallel/other_features

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 环境变量

   env/env_var_list
