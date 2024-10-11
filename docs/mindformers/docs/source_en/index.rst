MindSpore Transformers Documentation
=====================================

The goal of MindSpore Transformers (also known as MindFormers) suite is to build a full-process development suite for training, fine-tuning, evaluating, inference, and deploying large models, providing the industry mainstream Transformer class of pre-trained models and SOTA downstream task applications, and covering a rich range of parallel features, with the expectation of helping users to easily realize large model training and innovative research and development.

If you have any suggestions for MindFormers, please contact us via `issue <https://gitee.com/mindspore/mindformers/issues>`_ and we will handle them promptly.

MindFormers supports one-click start of single/multi-card training, fine-tuning, evaluation, and inference processes for any task, which makes the execution of deep learning tasks more efficient and user-friendly by simplifying the operation, providing flexibility, and automating the process. Users can learn from the following explanatory documents:

- `Development Migration <https://www.mindspore.cn/mindformers/docs/en/dev/usage/dev_migration.html>`_
- `Pretraining <https://www.mindspore.cn/mindformers/docs/en/dev/usage/pre_training.html>`_
- `SFT Tuning <https://www.mindspore.cn/mindformers/docs/en/dev/usage/sft_tuning.html>`_
- `Low-Parameter Fine-Tuning <https://www.mindspore.cn/mindformers/docs/en/dev/usage/parameter_efficient_fine_tune.html>`_
- `Evaluation <https://www.mindspore.cn/mindformers/docs/en/dev/usage/evaluation.html>`_
- `Inference <https://www.mindspore.cn/mindformers/docs/en/dev/usage/inference.html>`_
- `Quantization <https://www.mindspore.cn/mindformers/docs/en/dev/usage/quantization.html>`_
- `MindIE Service Deployment <https://www.mindspore.cn/mindformers/docs/en/dev/usage/mindie_deployment.html>`_

Flexible and Easy-to-Use Personalized Configuration with MindFormers
----------------------------------------------------------------------

With its powerful feature set, MindFormers provides users with flexible and easy-to-use personalized configuration options. Specifically, it comes with the following key features:

1. `Weight Slicing and Merging <https://www.mindspore.cn/mindformers/docs/en/dev/function/transform_weight.html>`_

   Weights in different distributed scenarios are flexibly sliced and merged.

2. `Distributed Parallel <https://www.mindspore.cn/mindformers/docs/en/dev/function/distributed_parallel.html>`_

   One-click configuration of multi-dimensional hybrid distributed parallel allows models to run efficiently in clusters up to 10,000 cards.

3. `Dataset <https://www.mindspore.cn/mindformers/docs/en/dev/function/dataset.html>`_

   Support multiple forms of datasets.

4. `Weight Saving and Resumable Training After Breakpoint <https://www.mindspore.cn/mindformers/docs/en/dev/function/resume_training.html>`_

   Supports step-level resumable training after breakpoint, effectively reducing the waste of time and resources caused by unexpected interruptions during large-scale training.

Deep Optimizing with MindFormers
------------------------------------

- `Precision Optimizing <https://www.mindspore.cn/mindformers/docs/en/dev/acc_optimize/acc_optimize.html>`_
- `Performance Optimizing <https://www.mindspore.cn/mindformers/docs/en/dev/perf_optimize/perf_optimize.html>`_

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
   usage/parameter_efficient_fine_tune
   usage/evaluation
   usage/inference
   usage/mindie_deployment
   usage/quantization

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