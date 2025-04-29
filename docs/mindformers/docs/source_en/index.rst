MindSpore Transformers Documentation
=====================================

MindSpore Transformers (also known as MindFormers) is a MindSpore-native foundation model suite designed to provide full-flow development capabilities for foundation model training, fine-tuning, evaluating, inference and deploying, providing the industry mainstream Transformer class of pre-trained models and SOTA downstream task applications, and covering a rich range of parallel features, with the expectation of helping users to easily realize large model training and innovative research and development.

Users can refer to `Overall Architecture <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/overview.html>`_ and `Model Library <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/models.html>`_ to get a quick overview of the MindSpore Transformers system architecture, and the list of supported functional features and foundation models. Further, refer to the `Installation <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/quick_start/install.html>`_ and `Quick Start <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/quick_start/source_code_start.html>`_ to get started with MindSpore Transformers.

If you have any suggestions for MindSpore Transformers, please contact us via `issue <https://gitee.com/mindspore/mindformers/issues>`_ and we will handle them promptly.

MindSpore Transformers supports one-click start of single/multi-card training, fine-tuning, evaluation, and inference processes for any task, which makes the execution of deep learning tasks more efficient and user-friendly by simplifying the operation, providing flexibility, and automating the process. Users can learn from the following explanatory documents:

- `Development Migration <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/dev_migration.html>`_
- `Pretraining <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/pre_training.html>`_
- `SFT Tuning <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/sft_tuning.html>`_
- `Evaluation <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/evaluation.html>`_
- `Inference <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/inference.html>`_
- `Quantization <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/quantization.html>`_
- `Service Deployment <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/mindie_deployment.html>`_
- `Multimodal Model Development <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/multi_modal.html>`_

Code repository address: <https://gitee.com/mindspore/mindformers>

Flexible and Easy-to-Use Personalized Configuration with MindSpore Transformers
-------------------------------------------------------------------------------------------

With its powerful feature set, MindSpore Transformers provides users with flexible and easy-to-use personalized configuration options. Specifically, it comes with the following key features:

1. `Weight Format Conversion <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/weight_conversion.html>`_

   Provides a unified weight conversion tool that converts model weights between the formats used by HuggingFace and MindSpore Transformers.

2. `Distributed Weight Slicing and Merging <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/transform_weight.html>`_

   Weights in different distributed scenarios are flexibly sliced and merged.

3. `Distributed Parallel <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/distributed_parallel.html>`_

   One-click configuration of multi-dimensional hybrid distributed parallel allows models to run efficiently in clusters up to 10,000 cards.

4. `Dataset <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/dataset.html>`_

   Support multiple types and formats of datasets.

5. `Weight Saving and Resumable Training After Breakpoint <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/resume_training.html>`_

   Supports step-level resumable training after breakpoint, effectively reducing the waste of time and resources caused by unexpected interruptions during large-scale training.

6. `Training Metrics Monitoring <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/monitor.html>`_

   Provides visualization services for the training phase of large models for monitoring and analyzing various indicators and information during the training process.

7. `Training High Availability <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/high_availability.html>`_

   Provide high-availability capabilities for the training phase of large models, including end-of-life CKPT preservation, UCE fault-tolerant recovery, and process-level rescheduling recovery.

8. `Safetensors Weights <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/safetensors.html>`_

   Support the function of saving and loading weight files in safetensors format.

9. `Fine-Grained Activations SWAP <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/fine_grained_activations_swap.html>_`

   Support fine-grained selection of specific activations to enable SWAP and reduce peak memory overhead during model training.

Deep Optimizing with MindSpore Transformers
---------------------------------------------

- `Precision Optimizing <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/acc_optimize/acc_optimize.html>`_
- `Performance Optimizing <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/perf_optimize/perf_optimize.html>`_

Appendix
------------------------------------

- `Environment Variables Descriptions <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/appendix/env_variables.html>`_
- `Configuration File Descriptions <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/appendix/conf_files.html>`_

FAQ
------------------------------------

- `Model-Related <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/faq/model_related.html>`_
- `Function-Related <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/faq/func_related.html>`_
- `MindSpore Transformers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/faq/mindformers_contribution.html>`_
- `Modelers Contribution Guide <https://www.mindspore.cn/mindformers/docs/en/r1.5.0/faq/modelers_contribution.html>`_

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
   usage/multi_modal
   usage/pre_training
   usage/sft_tuning
   usage/evaluation
   usage/inference
   usage/quantization
   usage/mindie_deployment
   usage/pretrain_gpt

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
   function/monitor
   function/high_availability
   function/safetensors
   function/fine_grained_activations_swap

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE
