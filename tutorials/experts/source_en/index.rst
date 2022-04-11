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

   data_engine/auto_augmentation
   data_engine/eager
   data_engine/cache
   data_engine/optimize_data_processing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Operator Execution

   operation/op_classification
   operation/op_overload
   operation/op_cpu
   operation/op_gpu
   operation/op_ascend
   operation/op_custom

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Inference

   model_infer/inference
   model_infer/online_inference
   model_infer/offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Debugging and Tuning

   debug/read_ir_files
   debug/debug_in_pynative_mode
   debug/dump_in_graph_mode
   debug/custom_debugging_info
   debug/incremental_compilation
   debug/auto_tune
   debug/dataset_autotune

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Parallel

   parallel/distributed_training
   parallel/distributed_advanced
   parallel/distributed_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Features

   others/mixed_precision
   others/gradient_accumulation
   others/second_order_optimizer
   others/evaluate_the_model_during_training
   others/on_device
   
   