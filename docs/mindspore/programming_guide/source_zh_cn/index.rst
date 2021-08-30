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
   :caption: 设计介绍

   design/technical_white_paper
   design/distributed_training_design
   design/mindir
   可视化调试调优设计↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/training_visual_design.html>
   design/glossary

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   实现简单线性函数拟合↗ <https://www.mindspore.cn/tutorials/zh-CN/master/linear_regression.html> 
   实现一个图片分类应用↗ <https://www.mindspore.cn/tutorials/zh-CN/master/quick_start.html>
   quick_start/quick_video

.. toctree::
   :maxdepth: 1
   :caption: 基本概念

   dtype
   tensor
   parameter_introduction
   operators
   cell

.. toctree::
   :maxdepth: 1
   :caption: 数据加载和处理

   dataset_sample
   dataset
   pipeline
   dataset_advanced
   
.. toctree::
   :maxdepth: 1
   :caption: 网络构建
   
   build_net
   initializer
   parameter
   loss
   grad_operation
   hypermap
   optim

.. toctree::
   :maxdepth: 1
   :caption: 运行管理

   context

.. toctree::
   :maxdepth: 1
   :caption: 模型运行

   run
   save_and_load_models
   model

.. toctree::
   :maxdepth: 1
   :caption: 推理

   multi_platform_inference
   online_inference
   offline_inference

.. toctree::
   :maxdepth: 1
   :caption: 分布式并行

   distributed_training
   distributed_training_ascend
   distributed_training_gpu
   apply_pipeline_parallel
   apply_host_device_training
   apply_parameter_server_training
   distributed_training_transformer
   pangu_alpha
   save_load_model_hybrid_parallel
   distributed_inference
   auto_parallel

.. toctree::
   :maxdepth: 1
   :caption: Numpy

   numpy

.. toctree::
   :maxdepth: 1
   :caption: 高级特性

   second_order_optimizer
   apply_quantization_aware_training

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

   accuracy_optimization

.. toctree::
   :maxdepth: 1
   :caption: 性能优化

   enable_mixed_precision
   enable_graph_kernel_fusion
   enable_auto_tune
   apply_gradient_accumulation
   使用Profiler调试性能↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 应用实践
   :hidden:

   cv
   nlp
   hpc
   use_on_the_cloud
