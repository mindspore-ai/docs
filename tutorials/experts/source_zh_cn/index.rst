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
   dataset/eager
   dataset/cache
   dataset/optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 算子执行

   operation/op_classification
   operation/op_overload
   operation/op_cpu
   operation/op_gpu
   operation/op_ascend
   operation/op_custom

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型推理

   infer/inference
   infer/cpu_gpu_mindir
   infer/ascend_910_mindir
   infer/ascend_310_mindir
   infer/ascend_310_air

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 调试调优

   debug/mindir
   debug/dump
   debug/custom_debug
   debug/ms_class
   debug/op_compilation
   debug/auto_tune
   debug/dataset_autotune
   debug/fixing_randomness
   debug/pynative

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行

   parallel/introduction
   parallel/communicate_ops
   parallel/train_ascend
   parallel/train_gpu
   parallel/distributed_inference
   parallel/save_load
   parallel/transformer
   parallel/pangu_alpha
   parallel/fault_recover

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 其他特性

   others/mixed_precision
   others/gradient_accumulation
   others/adaptive_summation
   others/dimention_reduce_training
   others/second_order_optimizer
   