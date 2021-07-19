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
   :caption: Quickstart

   quick_start/linear_regression
   quick_start/quick_start
   quick_start/quick_video

.. toctree::
   :maxdepth: 1
   :caption: Basic Concepts

   dtype
   tensor
   operators
   cell

.. toctree::
   :maxdepth: 1
   :caption: Numpy

   numpy

.. toctree::
   :maxdepth: 1
   :caption: Execution Management

   context

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
   layer
   loss
   optim
   custom_net
   network_component

.. toctree::
   :maxdepth: 1
   :caption: Train Models

   run
   callback
   save_model
   load_model_for_inference_and_transfer
   train

.. toctree::
   :maxdepth: 1
   :caption: Inference

   multi_platform_inference
   multi_platform_inference_ascend_910
   multi_platform_inference_ascend_310
   multi_platform_inference_gpu
   multi_platform_inference_cpu

.. toctree::
   :maxdepth: 1
   :caption: Distributed Training

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
   :caption: Function Debugging

   debug_in_pynative_mode
   dump_in_graph_mode
   custom_debugging_info
   evaluate_the_model_during_training
   incremental_operator_build

.. toctree::
   :maxdepth: 1
   :caption: Performance Optimization

   optimize_data_processing
   enable_mixed_precision
   enable_graph_kernel_fusion
   apply_gradient_accumulation
   apply_quantization_aware_training
   apply_post_training_quantization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Application
   :hidden:
 
   cv
   nlp
   hpc
   use_on_the_cloud
