.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

For Experts
=========================

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
   :caption: Graph Compilation

   network/control_flow
   network/op_overload
   network/jit_class
   network/constexpr
   network/dependency_control

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Training Optimization

   optimize/execution_opt
   optimize/gradient_accumulation
   optimize/adaptive_summation
   optimize/dimention_reduce_training
   optimize/thor

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
   :caption: Automatic Vectorization

   vmap/vmap

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Inference

   infer/inference
   infer/cpu_gpu_mindir
   infer/ascend_910_mindir
   infer/ascend_310_mindir
   infer/ascend_310_air
   infer/model_compression

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Debugging and Tuning
   :titlesonly:

   debug/function_debug
   debug/performance_optimization
   Precision Optimization↗ <https://mindspore.cn/mindinsight/docs/en/master/accuracy_problem_preliminary_location.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Parallel
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
   :caption: Environment Variables

   env/env_var_list
