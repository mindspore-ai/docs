MindSpore Transformers Documentation
=====================================

The goal of MindSpore Transformers suite is to build a full-process development suite for training, fine-tuning, evaluating, inference, and deploying large models, providing the industry mainstream Transformer class of pre-trained models and SOTA downstream task applications, and covering a rich range of parallel features, with the expectation of helping users to easily realize large model training and innovative research and development.

Users can refer to `Overall Architecture <https://www.mindspore.cn/mindformers/docs/en/dev/start/overview.html>`_ and `Model Library <https://www.mindspore.cn/mindformers/docs/en/dev/start/models.html>`_ to get an initial understanding of MindSpore Transformers architecture and model support. Refer to the `Installation <https://www.mindspore.cn/mindformers/docs/en/dev/quick_start/install.html>`_ and `Quick Start <https://www.mindspore.cn/mindformers/docs/en/dev/quick_start/source_code_start.html>`_ to get started with MindSpore Transformers.

If you have any suggestions for MindSpore Transformers, please contact us via `issue <https://gitee.com/mindspore/mindformers/issues>`_ and we will handle them promptly.

MindSpore Transformers supports one-click start of single/multi-card training, fine-tuning, evaluation, and inference processes for any task, which makes the execution of deep learning tasks more efficient and user-friendly by simplifying the operation, providing flexibility, and automating the process. Users can learn from the following explanatory documents:

- `Development Migration <https://www.mindspore.cn/mindformers/docs/en/dev/usage/dev_migration.html>`_
- `Pretraining <https://www.mindspore.cn/mindformers/docs/en/dev/usage/pre_training.html>`_
- `SFT Tuning <https://www.mindspore.cn/mindformers/docs/en/dev/usage/sft_tuning.html>`_
- `Evaluation <https://www.mindspore.cn/mindformers/docs/en/dev/usage/evaluation.html>`_
- `Inference <https://www.mindspore.cn/mindformers/docs/en/dev/usage/inference.html>`_
- `Quantization <https://www.mindspore.cn/mindformers/docs/en/dev/usage/quantization.html>`_
- `Service Deployment <https://www.mindspore.cn/mindformers/docs/en/dev/usage/mindie_deployment.html>`_
- `Multimodal Model Development <https://www.mindspore.cn/mindformers/docs/en/dev/usage/multi_modal.html>`_

Flexible and Easy-to-Use Personalized Configuration with MindSpore Transformers
-------------------------------------------------------------------------------------------

With its powerful feature set, MindSpore Transformers provides users with flexible and easy-to-use personalized configuration options. Specifically, it comes with the following key features:

1. `Weight Format Conversion <https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html>`_

   Provides a unified weight conversion tool that converts model weights between the formats used by HuggingFace and MindSpore Transformers.

2. `Distributed Weight Slicing and Merging <https://www.mindspore.cn/mindformers/docs/en/dev/function/transform_weight.html>`_

   Weights in different distributed scenarios are flexibly sliced and merged.

3. `Distributed Parallel <https://www.mindspore.cn/mindformers/docs/en/dev/function/distributed_parallel.html>`_

   One-click configuration of multi-dimensional hybrid distributed parallel allows models to run efficiently in clusters up to 10,000 cards.

4. `Dataset <https://www.mindspore.cn/mindformers/docs/en/dev/function/dataset.html>`_

   Support multiple forms of datasets.

5. `Weight Saving and Resumable Training After Breakpoint <https://www.mindspore.cn/mindformers/docs/en/dev/function/resume_training.html>`_

   Supports step-level resumable training after breakpoint, effectively reducing the waste of time and resources caused by unexpected interruptions during large-scale training.

Deep Optimizing with MindSpore Transformers
---------------------------------------------

- `Precision Optimizing <https://www.mindspore.cn/mindformers/docs/en/dev/acc_optimize/acc_optimize.html>`_
- `Performance Optimizing <https://www.mindspore.cn/mindformers/docs/en/dev/perf_optimize/perf_optimize.html>`_

Appendix
------------------------------------

- `Environment Variables Descriptions <https://www.mindspore.cn/mindformers/docs/en/dev/appendix/env_variables.html>`_
- `Configuration File Descriptions <https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html>`_

FAQ
------------------------------------

- `Model-Related <https://www.mindspore.cn/mindformers/docs/en/dev/faq/model_related.html>`_
- `Function-Related <https://www.mindspore.cn/mindformers/docs/en/dev/faq/func_related.html>`_
- `MindSpore Transformers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/dev/faq/mindformers_contribution.html>`_
- `Modelers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/dev/faq/modelers_contribution.html>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Start
   :hidden:

   start/overview
   start/models

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start/install
   quick_start/source_code_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Usage Tutorials
   :hidden:

   usage/dev_migration
   usage/pre_training
   usage/sft_tuning
   usage/evaluation
   usage/inference
   usage/quantization
   usage/mindie_deployment
   usage/pretrain_gpt
   usage/multi_modal

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Function Description
   :hidden:

   function/weight_conversion
   function/transform_weight
   function/distributed_parallel
   function/dataset
   function/resume_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Precision Optimization
   :hidden:

   acc_optimize/acc_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Performance Optimization
   :hidden:

   perf_optimize/perf_optimize

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   mindformers
   mindformers.core
   mindformers.dataset
   mindformers.generation
   mindformers.models
   mindformers.modules
   mindformers.pet
   mindformers.pipeline
   mindformers.tools
   mindformers.wrapper

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Appendix
   :hidden:

   appendix/env_variables
   appendix/conf_files

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ
   :hidden:

   faq/model_related
   faq/func_related
   faq/mindformers_contribution
   faq/modelers_contribution