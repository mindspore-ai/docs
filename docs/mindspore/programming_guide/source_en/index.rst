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
   design/distributed_training_design
   design/mindir
   Design of Visualization↗ <https://www.mindspore.cn/mindinsight/docs/en/master/training_visual_design.html>
   design/glossary

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   Implementing Simple Linear Function Fitting↗ <https://www.mindspore.cn/tutorials/en/master/linear_regression.html> 
   Implementing an Image Classification Application↗ <https://www.mindspore.cn/tutorials/en/master/quick_start.html>
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

   initializer
   parameter
   loss
   grad_operation
   constexpr
   hypermap
   optim

.. toctree::
   :maxdepth: 1
   :caption: Execution Management

   context

.. toctree::
   :maxdepth: 1
   :caption: Model Running

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

   debug_in_pynative_mode
   dump_in_graph_mode
   custom_debugging_info
   evaluate_the_model_during_training
   incremental_operator_build

.. toctree::
   :maxdepth: 1
   :caption: Performance Optimization

   enable_mixed_precision
   enable_graph_kernel_fusion
   enable_auto_tune
   apply_gradient_accumulation
   Debugging performance with Profiler↗ <https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling.html>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Application
   :hidden:
 
   cv
   nlp
   hpc
   use_on_the_cloud
