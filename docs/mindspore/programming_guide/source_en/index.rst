.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Programming Guide
=================================

.. toctree::
   :maxdepth: 1
   :caption: Overview

   architecture
   api_structure

.. toctree::
   :maxdepth: 1
   :caption: Design

   design/technical_white_paper
   design/gradient
   design/distributed_training_design
   design/mindir
   Design of Visualization↗ <https://www.mindspore.cn/mindinsight/docs/en/r1.5/training_visual_design.html>
   design/glossary

Note: Clicking on the title with "↗" will leave the Programming Guide page.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   Implementing Simple Linear Function Fitting↗ <https://www.mindspore.cn/tutorials/en/r1.5/linear_regression.html> 
   Implementing an Image Classification Application↗ <https://www.mindspore.cn/tutorials/en/r1.5/quick_start.html>
   quick_start/quick_video

.. toctree::
   :maxdepth: 1
   :caption: Basic Concepts

   dtype
   tensor
   parameter_introduction
   operators
   cell

.. toctree::
   :maxdepth: 1
   :caption: Data Pipeline

   dataset_sample
   dataset
   pipeline
   dataset_advanced

.. toctree::
   :maxdepth: 1
   :caption: Build the Network

   build_net
   initializer
   parameter
   loss
   grad_operation
   indefinite_parameter
   constexpr
   hypermap
   optim

.. toctree::
   :maxdepth: 1
   :caption: Model Running

   context
   run
   save_and_load_models
   model

.. toctree::
   :maxdepth: 1
   :caption: Inference

   multi_platform_inference
   online_inference
   offline_inference

.. toctree::
   :maxdepth: 1
   :caption: Distributed Training

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
   :caption: Advanced Features

   second_order_optimizer
   apply_quantization_aware_training

.. toctree::
   :maxdepth: 1
   :caption: Function Debugging

   read_ir_files
   Debugging in PyNative Mode↗ <https://www.mindspore.cn/docs/programming_guide/en/r1.5/debug_in_pynative_mode.html>
   dump_in_graph_mode
   custom_debugging_info
   incremental_operator_build

.. toctree::
   :maxdepth: 1
   :caption: Performance Optimization

   enable_mixed_precision
   enable_graph_kernel_fusion
   enable_auto_tune
   apply_gradient_accumulation
   Debugging performance with Profiler↗ <https://www.mindspore.cn/mindinsight/docs/en/r1.5/performance_profiling.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Application
   :hidden:
 
   cv
   nlp
   hpc
   use_on_the_cloud
