.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore编程指南
===================

.. toctree::
   :maxdepth: 1
   :caption: 整体介绍

   architecture
   api_structure

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   quick_start/quick_start
   quick_start/linear_regression
   quick_start/quick_video

.. toctree::
   :maxdepth: 1
   :caption: 数据类型

   dtype
   tensor

.. toctree::
   :maxdepth: 1
   :caption: 计算组件

   operators
   parameter
   cell
   network_component
   initializer
   numpy

.. toctree::
   :maxdepth: 1
   :caption: 数据管道

   dataset_loading
   sampler
   pipeline
   augmentation
   tokenizer
   dataset_conversion
   auto_augmentation
   cache
   data_sample

.. toctree::
   :maxdepth: 1
   :caption: 执行管理

   context
   run
   callback
   save_model
   load_model_for_inference_and_transfer
   train

.. toctree::
   :maxdepth: 1
   :caption: 推理

   multi_platform_inference
   multi_platform_inference_ascend_910
   multi_platform_inference_ascend_310
   multi_platform_inference_gpu
   multi_platform_inference_cpu

.. toctree::
   :maxdepth: 1
   :caption: 分布式并行

   distributed_training
   distributed_training_ascend
   distributed_training_gpu
   apply_pipeline_parallel
   apply_host_device_training
   apply_parameter_server_training
   save_load_model_hybrid_parallel
   distributed_inference
   auto_parallel

.. toctree::
   :maxdepth: 1
   :caption: 功能调试

   debug_in_pynative_mode
   dump_in_graph_mode
   custom_debugging_info
   evaluate_the_model_during_training
   incremental_operator_build

.. toctree::
   :maxdepth: 1
   :caption: 精度调优

   精度调优思路和方法 <https://www.mindspore.cn/docs/migration_guide/zh-CN/r1.3/accuracy_optimization.html>

.. toctree::
   :maxdepth: 1
   :caption: 性能优化

   optimize_data_processing
   enable_mixed_precision
   enable_graph_kernel_fusion
   apply_gradient_accumulation
   apply_quantization_aware_training
   apply_post_training_quantization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 应用实践
   :hidden:

   cv
   nlp
   hpc
   use_on_the_cloud
