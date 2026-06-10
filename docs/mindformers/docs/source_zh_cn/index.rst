MindSpore Transformers 文档
=========================================

MindSpore Transformers 套件的目标是构建一个大模型预训练、微调、推理、部署的全流程开发套件，提供业内主流的 Transformer 类大语言模型（Large Language Models, LLMs）和多模态理解模型（Multimodal Models, MMs）。期望帮助用户轻松地实现大模型全流程开发。

.. note::

   自 **r2.0.0** 起，MindSpore Transformers 以 **动态图（PyNative）实现** 作为演进主线，文档默认面向动态图。原有 **静态图（GRAPH_MODE）实现** 的资料整体迁入 `静态图实现 <static_graph/introduction/overview.html>`_ 章节并标记为废弃；推理、服务化部署、量化等动态图尚未覆盖的能力，请前往该章节查阅。

   动态图文档正分批上线：未上线页面在正文中以「文档名」纯文本标注，正文将随后续提交上线；贡献指南与 FAQ 沿用原有页面。

MindSpore Transformers 的开源仓库地址为 `AtomGit | MindSpore/mindformers <https://atomgit.com/mindspore/mindformers>`_ 。如有任何建议，请通过 `issue <https://atomgit.com/mindspore/mindformers/issues>`_ 与我们联系。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 介绍
   :hidden:

   quick_start/quick_start
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
   :caption: 训练指南
   :hidden:

   guide/training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能特性
   :hidden:

   feature/overview
   feature/start_task
   feature/configuration
   feature/logging
   feature/dataset
   feature/training_hyperparameters
   feature/parallel_training
   feature/memory_optimization
   feature/save_and_load_checkpoint
   feature/resume_training
   feature/monitor
   feature/other_training_features
   feature/static_graph_features

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 静态图实现（Deprecated）
   :hidden:

   static_graph/introduction/overview
   static_graph/guide/index
   static_graph/feature/index
   static_graph/advanced_development/index
   static_graph/example/index
   static_graph/env_variables
