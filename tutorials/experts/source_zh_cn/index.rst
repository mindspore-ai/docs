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

   data_engine/auto_augmentation
   data_engine/eager
   data_engine/cache
   data_engine/optimize_data_processing

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

   model_infer/inference
   model_infer/online_inference
   model_infer/offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 调试调优

   debug/read_ir_files
   debug/debug_in_pynative_mode
   debug/dump_in_graph_mode
   debug/custom_debugging_info
   debug/incremental_compilation
   debug/auto_tune
   debug/dataset_autotune
   debug/ms_class

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行

   parallel/distributed_training
   parallel/distributed_advanced
   parallel/distributed_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高级特性

   others/mixed_precision
   others/gradient_accumulation
   
   