MindSpore Transformers Documentation
=====================================

The goal of the MindSpore Transformers suite is to build a full-process development suite for Large model pre-training, fine-tuning, inference, and deployment. It provides mainstream Transformer-based Large Language Models (LLMs) and Multimodal Models (MMs). It is expected to help users easily realize the full process of large model development.

.. note::

   Starting with **r2.0.0**, MindSpore Transformers has adopted a **dynamic graph (PyNative) implementation** as its primary development path, and the documentation is now primarily focused on dynamic graphs. All documentation related to the original **static graph (GRAPH_MODE) implementation** has been moved to the `Static Graph Implementation <static_graph/introduction/overview.html>`_ section and marked as deprecated. For information on capabilities not yet covered by dynamic graphs, such as inference, service-oriented deployment, and quantization, please refer to that section.

The open-source code repository for MindSpore Transformers is located at `AtomGit | MindSpore/mindformers <https://atomgit.com/mindspore/mindformers>`_. If you have any suggestions for MindSpore Transformers, please contact us via `issue <https://atomgit.com/mindspore/mindformers/issues>`_ and we will handle them promptly.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   quick_start/quick_start
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
   :caption: Training Guide
   :hidden:

   guide/training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Function Features
   :hidden:

   feature/overview
   feature/start_task
   feature/configuration
   feature/logging
   feature/dataset
   feature/training_hyperparameters
   feature/parallel_training
   feature/memory_optimization
   feature/save_load_checkpoint
   feature/resume_training
   feature/monitor
   feature/other_training_features
   feature/static_graph_features

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Static Graph Implementation (Deprecated)
   :hidden:

   static_graph/introduction/overview
   static_graph/guide/index
   static_graph/feature/index
   static_graph/advanced_development/index
   static_graph/example/index
   static_graph/env_variables

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE   