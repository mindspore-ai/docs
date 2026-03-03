MindSpore Transformers 文档
=========================================

MindSpore Transformers套件的目标是构建一个大模型预训练、微调、推理、部署的全流程开发套件，提供业内主流的Transformer类大语言模型（Large Language Models, LLMs）和多模态理解模型（Multimodal Models, MMs）。期望帮助用户轻松地实现大模型全流程开发。

MindSpore Transformers套件基于MindSpore内置的多维混合并行技术和组件化设计，具备如下特点：

- 一键启动模型单卡或多卡预训练、微调、推理、部署流程；
- 提供丰富的多维混合并行能力可供灵活易用地进行个性化配置；
- 大模型训推系统级深度优化，原生支持超大规模集群高效训推，故障快速恢复；
- 支持任务组件配置化开发。任意模块可通过统一配置进行使能，包括模型网络、优化器、学习率策略等；
- 提供训练精度/性能监控指标实时可视化能力等。

用户可以参阅 `整体架构 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/overview.html>`_ 和 `模型库 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/models.html>`_ ，快速了解MindSpore Transformers的系统架构，以及所支持的大模型清单。

MindSpore Transformers的开源仓库地址为 `AtomGit | MindSpore/mindformers <https://atomgit.com/mindspore/mindformers>`_ 。

如果您对MindSpore Transformers有任何建议，请通过 `issue <https://atomgit.com/mindspore/mindformers/issues>`_ 与我们联系，我们将及时处理。

使用MindSpore Transformers进行大模型全流程开发
-----------------------------------------------------

MindSpore Transformers 提供统一的一键启动脚本，支持单卡/多卡训练、微调与推理。从入门到上线，可按需查阅：`训练指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/llm_training.html>`_、`预训练实践 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/pre_training.html>`_、`监督微调实践 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/supervised_fine_tuning.html>`_、`推理指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/inference.html>`_、`服务化部署指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/deployment.html>`_ 与 `评测指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/evaluation.html>`_。


MindSpore Transformers 功能特性说明
-----------------------------------------------------

预训练、微调与推理全流程中的通用能力、训练能力（如数据集、并行、断点续训、内存优化等）以及推理与量化能力，均在 `功能特性概述 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/overview.html>`_ 中按类汇总，可从中快速查找并跳转到对应说明文档。

使用 MindSpore Transformers 进行高阶开发
--------------------------------------

在完成基础训练与推理后，若需进行模型迁移、精度与性能调优或与标杆做精度对比，可参阅 `高阶开发概述 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/overview.html>`_，其中按调试调优、模型开发与配置、精度对比及 API 参考分类整理了全部高阶开发文档。

环境变量、贡献与常见问题
------------------------------------

- 运行与调试相关环境变量见 `环境变量说明 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/env_variables.html>`_。
- 参与开发可参考 `MindSpore Transformers 贡献指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/contribution/mindformers_contribution.html>`_ 与 `魔乐社区贡献指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/contribution/modelers_contribution.html>`_。
- 常见问题见 `模型相关 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/faq/model_related.html>`_ 与 `功能相关 <https://www.mindspore.cn/mindformers/docs/zh-CN/master/faq/feature_related.html>`_ FAQ。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 介绍
   :hidden:

   introduction/overview
   introduction/models

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装
   :hidden:

   installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 大模型全流程指南
   :hidden:

   guide/llm_training
   guide/pre_training
   guide/supervised_fine_tuning
   guide/inference
   guide/deployment
   guide/evaluation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能特性
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
   :caption: 高阶开发
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
   :caption: 优秀实践
   :hidden:

   example/docker-installation
   example/distilled/distilled
   example/convert_ckpt_to_megatron/convert_ckpt_to_megatron
   example/model_test/model_test
   example/finetune_with_glm4/finetune_with_glm4

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 环境变量
   :hidden:

   env_variables

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 贡献指南
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
