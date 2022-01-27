.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Programming Guide
=================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Overview
   
   architecture
   api_structure

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Design
   
   design/technical_white_paper
   design/gradient
   design/distributed_training_design
   design/mindir
   Design of Visualization↗ <https://www.mindspore.cn/mindinsight/docs/en/r1.6/training_visual_design.html>
   design/glossary

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quickstart
   
   Implementing Simple Linear Function Fitting↗ <https://www.mindspore.cn/tutorials/en/r1.6/linear_regression.html>
   Implementing an Image Classification Application↗ <https://www.mindspore.cn/tutorials/en/r1.6/quick_start.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Basic Concepts
   
   dtype
   tensor
   parameter_introduction
   operators
   cell
   dataset_introduction

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Data Pipeline
   
   dataset_sample
   dataset
   pipeline
   dataset_advanced
   dataset_usage

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Build the Network
   
   build_net
   initializer
   parameter
   control_flow
   indefinite_parameter
   loss
   grad_operation
   constexpr
   hypermap
   optim

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Model Running
   
   context
   run
   save_and_load_models
   model

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Inference
   
   multi_platform_inference
   online_inference
   offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Training
   
   distributed_training
   distributed_advanced
   distributed_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: PyNative
   
   debug_in_pynative_mode

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Numpy
   
   numpy

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Function Debugging
   
   read_ir_files
   Debugging in PyNative Mode↗ <https://www.mindspore.cn/docs/programming_guide/en/r1.6/debug_in_pynative_mode.html>
   dump_in_graph_mode
   custom_debugging_info
   incremental_operator_build

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Performance Optimization
   
   enable_mixed_precision
   enable_auto_tune
   enable_dataset_autotune
   enable_dataset_offload
   apply_gradient_accumulation
   Debugging performance with Profiler↗ <https://www.mindspore.cn/mindinsight/docs/en/r1.6/performance_profiling.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Features
   
   second_order_optimizer
   graph_kernel_fusion
   apply_quantization_aware_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Application
   
   cv
   nlp
   hpc
   use_on_the_cloud
