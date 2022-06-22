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
   dataset/eager
   dataset/cache
   dataset/optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: network

   network/op_overload
   network/custom_cell_reverse
   network/ms_class

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Custom Operator

   operation/op_ascend
   operation/op_custom

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Inference

   infer/inference
   infer/cpu_gpu_mindir
   infer/ascend_910_mindir
   infer/ascend_310_mindir
   infer/ascend_310_air

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Debugging and Tuning

   debug/performance_optimization
   Precision Optimization↗ <https://mindspore.cn/mindinsight/docs/en/master/accuracy_problem_preliminary_location.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Parallel

   parallel/introduction
   parallel/train_ascend
   parallel/train_gpu
   parallel/distributed_inference
   parallel/save_load

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Features

   others/mixed_precision
   others/gradient_accumulation
   others/adaptive_summation
   others/dimention_reduce_training
   others/ms_operator
   