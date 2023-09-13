.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

For Experts
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Parallel
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
   parallel/platform_differences
   parallel/others
   parallel/distributed_case

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Custom Operator

   operation/op_custom
   operation/ms_kernel
   operation/op_custom_adv
   operation/op_custom_aot

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Performance Optimization

   Profiling↗ <https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling.html>
   optimize/execution_opt
   optimize/op_overload
   optimize/graph_fusion_engine
   optimize/op_compilation
   optimize/mem_reuse

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Algorithm Optimization

   optimize/gradient_accumulation
   optimize/thor

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: High-level Functional Programming

   vmap/vmap
   func_programming/Jacobians_Hessians
   func_programming/per_sample_gradients

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Data Processing

   dataset/augment
   dataset/cache
   dataset/optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Inference

   infer/inference
   infer/model_compression

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Complex Problem Debugging

   debug/dump
   debug/aoe
   debug/rdr
   debug/fault_recover