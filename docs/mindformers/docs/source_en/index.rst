MindSpore Transformers Documentation
=====================================

The goal of the MindSpore Transformers suite is to build a full-process development suite for Large model pre-training, fine-tuning, inference, and deployment. It provides mainstream Transformer-based Large Language Models (LLMs) and Multimodal Models (MMs). It is expected to help users easily realize the full process of large model development.

Based on MindSpore's built-in parallel technology and component-based design, the MindSpore Transformers suite has the following features:

- One-click initiation of single or multi-card pre-training, fine-tuning, inference, and deployment processes for large models;
- Provide rich multi-dimensional hybrid parallel capabilities for flexible and easy-to-use personalized configuration;
- System-level deep optimization on large model training and inference, native support for ultra-large-scale cluster efficient training and inference, rapid fault recovery;
- Support for configurable development of task components. Any module can be enabled by unified configuration, including model network, optimizer, learning rate policy, etc.;
- Provide real-time visualization of training accuracy/performance monitoring indicators.

Users can refer to `Overall Architecture <https://www.mindspore.cn/mindformers/docs/en/master/introduction/overview.html>`_ and `Model Library <https://www.mindspore.cn/mindformers/docs/en/master/introduction/models.html>`_ to get a quick overview of the MindSpore Transformers system architecture, and the list of supported foundation models.

The open-source code repository for MindSpore Transformers is located at `AtomGit | MindSpore/mindformers <https://atomgit.com/mindspore/mindformers>`_.

If you have any suggestions for MindSpore Transformers, please contact us via `issue <https://atomgit.com/mindspore/mindformers/issues>`_ and we will handle them promptly.

Full-process Developing with MindSpore Transformers
-------------------------------------------------------------------------------------------

MindSpore Transformers provides a unified one-click start for single- and multi-card training, fine-tuning, and inference. From getting started to going live, refer as needed to: `Training Guide <https://www.mindspore.cn/mindformers/docs/en/master/guide/llm_training.html>`_, `Pretraining <https://www.mindspore.cn/mindformers/docs/en/master/guide/pre_training.html>`_, `Supervised Fine-Tuning <https://www.mindspore.cn/mindformers/docs/en/master/guide/supervised_fine_tuning.html>`_, `Inference <https://www.mindspore.cn/mindformers/docs/en/master/guide/inference.html>`_, `Service Deployment <https://www.mindspore.cn/mindformers/docs/en/master/guide/deployment.html>`_, and `Evaluation <https://www.mindspore.cn/mindformers/docs/en/master/guide/evaluation.html>`_.

Features description of MindSpore Transformers
-------------------------------------------------------------------------------------------

General capabilities, training capabilities (such as dataset, parallelism, resumable training, memory optimization, etc.), and inference and quantization are summarized by category in the `Features Overview <https://www.mindspore.cn/mindformers/docs/en/master/feature/overview.html>`_. Use it to quickly find and jump to the right documentation.

Advanced developing with MindSpore Transformers
-------------------------------------------------

After you have basic training and inference in place, for model migration, precision and performance tuning, or accuracy comparison with a reference implementation, see the `Advanced Development Overview <https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/overview.html>`_, which organizes all advanced development docs by diagnostics and optimization, model development and configuration, accuracy comparison, and API reference.

Environment variables, contribution, and FAQ
------------------------------------

- For environment variables used in running and debugging, see `Environment Variables Description <https://www.mindspore.cn/mindformers/docs/en/master/env_variables.html>`_.
- To contribute, refer to the `MindSpore Transformers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/master/contribution/mindformers_contribution.html>`_ and the `Modelers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/master/contribution/modelers_contribution.html>`_.
- For common issues, see the `Model-Related <https://www.mindspore.cn/mindformers/docs/en/master/faq/model_related.html>`_ and `Function-Related <https://www.mindspore.cn/mindformers/docs/en/master/faq/feature_related.html>`_ FAQ.

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
   guide/evaluation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Features
   :hidden:

   feature/overview
   feature/start_tasks
   feature/ckpt
   feature/safetensors
   feature/configuration
   feature/load_huggingface_config
   feature/logging
   feature/tokenizer
   feature/dataset
   feature/training_hyperparameters
   feature/monitor
   feature/resume_training
   feature/checkpoint_saving_and_loading
   feature/resume_training2.0
   feature/parallel_training
   feature/high_availability
   feature/memory_optimization
   feature/skip_data_and_ckpt_health_monitor
   feature/pma_fused_checkpoint
   feature/other_training_features
   feature/quantization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Development
   :hidden:

   advanced_development/overview
   advanced_development/precision_optimization
   advanced_development/performance_optimization
   advanced_development/dev_migration
   advanced_development/yaml_config_inference
   advanced_development/inference_precision_comparison
   advanced_development/accuracy_comparison
   advanced_development/training_template_instruction
   advanced_development/weight_transfer
   advanced_development/api

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Excellent Practice
   :hidden:

   example/distilled/distilled

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