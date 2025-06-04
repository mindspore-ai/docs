MindSpore Transformers Documentation
=====================================

The goal of the MindSpore Transformers suite is to build a full-process development suite for Large model pre-training, fine-tuning, inference, and deployment. It provides mainstream Transformer-based Large Language Models (LLMs) and Multimodal Models (MMs). It is expected to help users easily realize the full process of large model development.

Based on MindSpore's built-in parallel technology and component-based design, the MindSpore Transformers suite has the following features:

- One-click initiation of single or multi card pre-training, fine-tuning, inference, and deployment processes for large models;
- Provides rich multi-dimensional hybrid parallel capabilities for flexible and easy-to-use personalized configuration;
- System-level deep optimization on large model training and inference, native support for ultra-large-scale cluster efficient training and inference, rapid fault recovery;
- Support for configurable development of task components. Any module can be enabled by unified configuration, including model network, optimizer, learning rate policy, etc.;
- Provide real-time visualization of training accuracy/performance monitoring indicators.

Users can refer to `Overall Architecture <https://www.mindspore.cn/mindformers/docs/en/dev/introduction/overview.html>`_ and `Model Library <https://www.mindspore.cn/mindformers/docs/en/dev/introduction/models.html>`_ to get a quick overview of the MindSpore Transformers system architecture, and the list of supported foundation models.

If you have any suggestions for MindSpore Transformers, please contact us via `issue <https://gitee.com/mindspore/mindformers/issues>`_ and we will handle them promptly.

Full-process Developing with MindSpore Transformers
-------------------------------------------------------------------------------------------

MindSpore Transformers supports one-click start of single/multi-card training, fine-tuning, and inference processes for any task, which makes the execution of deep learning tasks more efficient and user-friendly by simplifying the operation, providing flexibility, and automating the process. Users can learn from the following explanatory documents:

- `Pretraining <https://www.mindspore.cn/mindformers/docs/en/dev/guide/pre_training.html>`_
- `Supervised Fine-Tuning <https://www.mindspore.cn/mindformers/docs/en/dev/guide/supervised_fine_tuning.html>`_
- `Inference <https://www.mindspore.cn/mindformers/docs/en/dev/guide/inference.html>`_
- `Service Deployment <https://www.mindspore.cn/mindformers/docs/en/dev/guide/deployment.html>`_

Code repository address: <https://gitee.com/mindspore/mindformers>

Features description of MindSpore Transformers
-------------------------------------------------------------------------------------------

MindSpore Transformers provides a wealth of features throughout the full-process of large model development. Users can learn about these features via the following links:

- General Features:

  - `Start Tasks <https://www.mindspore.cn/mindformers/docs/en/dev/feature/start_tasks.html>`_

    One-click start for single-device, single-node and multi-node tasks.

  - `Weight Format Conversion <https://www.mindspore.cn/mindformers/docs/en/dev/feature/weight_conversion.html>`_

    Provides a unified weight conversion tool that converts model weights between the formats used by HuggingFace and MindSpore Transformers.

  - `Distributed Weight Slicing and Merging <https://www.mindspore.cn/mindformers/docs/en/dev/feature/transform_weight.html>`_

    Weights in different distributed scenarios are flexibly sliced and merged.

  - `Safetensors Weights <https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html>`_

    Supports saving and loading weight files in safetensors format.

  - `Configuration File <https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html>`_

    Supports the use of `YAML` files to centrally manage and adjust configurable items in tasks.

  - `Logging <https://www.mindspore.cn/mindformers/docs/en/dev/feature/logging.html>`_

    Introduction of logs, including log structure, log saving, and so on.

- Training Features:

  - `Dataset <https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html>`_

    Supports multiple types and formats of datasets.

  - `Model Training Hyperparameters <https://www.mindspore.cn/mindformers/docs/en/dev/feature/training_hyperparameters.html>`_

    Flexibly configure hyperparameter settings for large model training.

  - `Training Metrics Monitoring <https://www.mindspore.cn/mindformers/docs/en/dev/feature/monitor.html>`_

    Provides visualization services for the training phase of large models for monitoring and analyzing various indicators and information during the training process.

  - `Resumable Training After Breakpoint <https://www.mindspore.cn/mindformers/docs/en/dev/feature/resume_training.html>`_

    Supports step-level resumable training after breakpoint, effectively reducing the waste of time and resources caused by unexpected interruptions during large-scale training.

  - `Training High Availability (Beta) <https://www.mindspore.cn/mindformers/docs/en/dev/feature/high_availability.html>`_

    Provides high-availability capabilities for the training phase of large models, including end-of-life CKPT preservation, UCE fault-tolerant recovery, and process-level rescheduling recovery (Beta feature).

  - `Parallel Training <https://www.mindspore.cn/mindformers/docs/en/dev/feature/parallel_training.html>`_

    One-click configuration of multi-dimensional hybrid distributed parallel allows models to run efficiently in clusters up to 10,000 cards.

  - `Training Memory Optimization <https://www.mindspore.cn/mindformers/docs/en/dev/feature/memory_optimization.html>`_

    Supports fine-grained recomputation and activations swap, to reduce peak memory overhead during model training.

  - `Other Training Features <https://www.mindspore.cn/mindformers/docs/en/dev/feature/other_training_features.html>`_

    Supports gradient accumulation and gradient clipping, etc.

- Inference Features:

  - `Evaluation <https://www.mindspore.cn/mindformers/docs/en/dev/feature/evaluation.html>`_

    Supports the use of third-party open-source evaluation frameworks and datasets for large-scale model ranking evaluations.

  - `Quantization <https://www.mindspore.cn/mindformers/docs/en/dev/feature/quantization.html>`_

    Integrated MindSpore Golden Stick toolkit to provides a unified quantization inference process.

Advanced developing with MindSpore Transformers
-------------------------------------------------

- Diagnostics and Optimization

  - `Precision Optimization <https://www.mindspore.cn/mindformers/docs/en/dev/advanced_development/precision_optimization.html>`_
  - `Performance Optimization <https://www.mindspore.cn/mindformers/docs/en/dev/advanced_development/performance_optimization.html>`_

- Model Development

  - `Development Migration <https://www.mindspore.cn/mindformers/docs/en/dev/advanced_development/dev_migration.html>`_
  - `Multimodal Model Development <https://www.mindspore.cn/mindformers/docs/en/dev/advanced_development/multi_modal_dev.html>`_

Environment Variables
------------------------------------

- `Environment Variables Description <https://www.mindspore.cn/mindformers/docs/en/dev/env_variables.html>`_

Contribution Guide
------------------------------------

- `MindSpore Transformers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/dev/contribution/mindformers_contribution.html>`_
- `Modelers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/dev/contribution/modelers_contribution.html>`_

FAQ
------------------------------------

- `Model-Related <https://www.mindspore.cn/mindformers/docs/en/dev/faq/model_related.html>`_
- `Function-Related <https://www.mindspore.cn/mindformers/docs/en/dev/faq/feature_related.html>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   introduction/overview
   introduction/models

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation
   :hidden:

   installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Full-process Guide to Large Models
   :hidden:

   guide/pre_training
   guide/supervised_fine_tuning
   guide/inference
   guide/deployment

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Features
   :hidden:

   feature/start_tasks
   feature/weight_conversion
   feature/transform_weight
   feature/safetensors
   feature/configuration
   feature/logging
   feature/training_function
   feature/infer_function

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Development
   :hidden:

   advanced_development/precision_optimization
   advanced_development/performance_optimization
   advanced_development/dev_migration
   advanced_development/multi_modal_dev
   advanced_development/api

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Environment Variables
   :hidden:

   env_variables

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Contribution Guide
   :hidden:

   contribution/mindformers_contribution
   contribution/modelers_contribution

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ
   :hidden:

   faq/model_related
   faq/feature_related