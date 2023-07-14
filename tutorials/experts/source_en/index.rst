.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

For Experts
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Static Graph Usage Sepecifications

   network/jit_fallback
   network/control_flow
   network/jit_class
   network/constexpr
   network/dependency_control

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Parallel
   :titlesonly:

   parallel/overview
   parallel/basic_cases
   parallel/operator_parallel
   parallel/pipeline_parallel
   parallel/optimizer_parallel
   parallel/recompute
   parallel/host_device_training
   parallel/parameter_server_training
   parallel/startup_method
   parallel/distributed_inference
   parallel/distributed_case

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Custom Operator

   operation/op_custom
   operation/ms_kernel
   operation/op_custom_adv

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
   optimize/adaptive_summation
   optimize/dimention_reduce_training
   optimize/thor

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: High-level Functional Programming

   vmap/vmap

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