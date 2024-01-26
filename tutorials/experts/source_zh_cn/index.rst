.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

深度开发
=====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行
   :titlesonly:

   parallel/overview
   parallel/startup_method
   parallel/data_parallel
   parallel/semi_auto_parallel
   parallel/auto_parallel
   parallel/manual_parallel
   parallel/parameter_server_training
   parallel/model_save_load
   parallel/recover
   parallel/optimize_technique
   parallel/others
   parallel/distributed_case

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 自定义算子

   operation/op_custom
   operation/ms_kernel
   operation/op_custom_adv
   operation/op_custom_aot

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 性能优化

   Profiling↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling.html>
   optimize/execution_opt
   optimize/graph_fusion_engine
   optimize/mem_reuse

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 算法优化

   optimize/gradient_accumulation
   optimize/thor

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高阶函数式编程

   vmap/vmap
   func_programming/Jacobians_Hessians
   func_programming/per_sample_gradients

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
   :caption: 模型推理

   infer/inference
   infer/model_compression

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 复杂问题调试

   debug/dump
   debug/aoe
   debug/rdr
   debug/fault_recover
