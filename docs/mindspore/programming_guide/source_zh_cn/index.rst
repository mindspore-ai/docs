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
   design/all_scenarios_architecture
   design/gradient
   design/dynamic_graph_and_static_graph
   design/distributed_training_design
   design/heterogeneous_training
   design/mindir
   design/data_engine
   可视化调试调优↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/training_visual_design.html>
   安全可信↗ <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/design.html>
   design/glossary

注意：点击带“↗”的标题，将会离开编程指南页面。

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
   dataset_introduction

.. toctree::
   :maxdepth: 1
   :caption: 数据加载和处理

   dataset_sample
   dataset
   pipeline
   dataset_advanced
   dataset_usage

.. toctree::
   :maxdepth: 1
   :caption: 网络构建

   build_net
   initializer
   parameter
   control_flow
   indefinite_parameter
   constexpr
   loss
   grad_operation
   hypermap
   optim
   train_and_eval

.. toctree::
   :maxdepth: 1
   :caption: 模型运行

   context
   run
   ms_function
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
   distributed_advanced
   distributed_example

.. toctree::
   :maxdepth: 1
   :caption: PyNative

   debug_in_pynative_mode

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

   read_ir_files
   使用PyNative模式调试↗ <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/debug_in_pynative_mode.html>
   dump_in_graph_mode
   custom_debugging_info
   incremental_operator_build
   fixing_randomness

.. toctree::
   :maxdepth: 1
   :caption: 精度调优

   精度问题初步定位指南↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_problem_preliminary_location.html>
   精度问题详细定位和调优指南↗ <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_optimization.html>

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
