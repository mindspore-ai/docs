MindSpore Transformers 文档
=========================================

MindSpore Transformers套件的目标是构建一个大模型预训练、微调、推理、部署的全流程开发套件，提供业内主流的Transformer类大语言模型（Large Language Models, LLMs）和多模态理解模型（Multimodal Models, MMs）。期望帮助用户轻松地实现大模型全流程开发。

MindSpore Transformers套件基于MindSpore内置的多维混合并行技术和组件化设计，具备如下特点：

- 一键启动模型单卡或多卡预训练、微调、推理、部署流程；
- 提供丰富的多维混合并行能力可供灵活易用地进行个性化配置；
- 大模型训推系统级深度优化，原生支持超大规模集群高效训推，故障快速恢复；
- 支持任务组件配置化开发。任意模块可通过统一配置进行使能，包括模型网络、优化器、学习率策略等；
- 提供训练精度/性能监控指标实时可视化能力等。

用户可以参阅 `整体架构 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/introduction/overview.html>`_ 和 `模型库 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/introduction/models.html>`_ ，快速了解MindSpore Transformers的系统架构，以及所支持的大模型清单。

如果您对MindSpore Transformers有任何建议，请通过 `issue <https://gitee.com/mindspore/mindformers/issues>`_ 与我们联系，我们将及时处理。

使用MindSpore Transformers进行大模型全流程开发
-----------------------------------------------------

MindSpore Transformers提供了统一的一键启动脚本，支持一键启动任意任务的单卡/多卡训练、微调、推理流程，它通过简化操作、提供灵活性和自动化流程，使得深度学习任务的执行变得更加高效和用户友好，用户可以通过以下说明文档进行学习：

.. raw:: html

   <table style="width: 100%">
      <tr>
         <td style="text-align: center; width: 20%; border: none">
            <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/full-process_1.png">
         </td>
         <td style="text-align: center; width: 20%; border: none">
            <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/full-process_2.png">
         </td>
         <td style="text-align: center; width: 20%; border: none">
            <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/full-process_3.png">
         </td>
      </tr>
      <tr>
         <td style="text-align: center; width: 20%; border: none">
            <ul style="text-align: left; display: inline-block;">
                <li><a href="https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/pre_training.html"><span>预训练</span></a></li>
            </ul>
         </td>
         <td style="text-align: center; width: 20%; border: none">
            <ul style="text-align: left; display: inline-block;">
                <li><a href="https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/supervised_fine_tuning.html"><span>微调</span></a></li>
            </ul>
         </td>
         <td style="text-align: center; width: 20%; border: none">
            <ul style="text-align: left; display: inline-block;">
                <li><a href="https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/inference.html"><span>推理</span></a></li>
                <li><a href="https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/deployment.html"><span>部署</span></a></li>
            </ul>
         </td>
      </tr>
   </table>

代码仓地址： <https://gitee.com/mindspore/mindformers>

MindSpore Transformers功能特性说明
-----------------------------------------------------




- 通用功能：

  - `启动任务 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/start_tasks.html>`_

    单卡、单机和多机任务一键启动。

  - `Ckpt权重 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/ckpt.html>`_

    支持ckpt格式的权重文件转换及切分功能。

  - `Safetensors权重 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html>`_

    支持safetensors格式的权重文件保存及加载功能。

  - `配置文件 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html>`_

    支持使用`YAML`文件集中管理和调整任务中的可配置项。

  - `日志 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/logging.html>`_

    日志相关介绍，包括日志结构、日志保存等。

- 训练功能：

  - `数据集 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html>`_

    支持多种类型和格式的数据集。

  - `训练超参数 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/training_hyperparameters.html>`_

    灵活配置大模型训练的超参数配置。

  - `训练指标监控 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/monitor.html>`_

    提供大模型训练阶段的可视化服务，用于监控和分析训练过程中的各种指标和信息。

  - `断点续训 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/resume_training.html>`_

    支持step级断点续训，有效减少大规模训练时意外中断造成的时间和资源浪费。

  - `训练高可用（Beta） <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/high_availability.html>`_

    提供大模型训练阶段的高可用能力，包括临终 CKPT 保存、UCE 故障容错恢复和进程级重调度恢复功能（Beta特性）。

  - `分布式训练 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/parallel_training.html>`_

    一键配置多维混合分布式并行，让模型在上至万卡的集群中高效训练。

  - `训练内存优化 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/memory_optimization.html>`_

    支持细粒度选择重计算和细粒度激活值SWAP，用于降低模型训练的峰值内存开销。

  - `其它训练特性 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/other_training_features.html>`_

    支持梯度累积、梯度裁剪等特性。

- 推理功能

  - `评测 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/evaluation.html>`_

    支持使用第三方开源评测框架和数据集进行大模型榜单评测。

  - `量化 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/quantization.html>`_

    集成 MindSpore Golden Stick 工具组件，提供统一量化推理流程开箱即用。

使用MindSpore Transformers进行高阶开发
--------------------------------------

- 调试调优

  - `精度调优 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/precision_optimization.html>`_
  - `性能调优 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/performance_optimization.html>`_

- 模型开发

  - `开发迁移 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/dev_migration.html>`_
  - `多模态理解模型开发 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/multi_modal_dev.html>`_

- 精度对比

  - `Parallel Core精度对比 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/accuracy_comparison.html>`_

环境变量
------------------------------------

- `环境变量说明 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/env_variables.html>`_

贡献指南
------------------------------------

- `MindSpore Transformers贡献指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/contribution/mindformers_contribution.html>`_
- `魔乐社区贡献指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/contribution/modelers_contribution.html>`_

FAQ
------------------------------------

- `模型相关 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/model_related.html>`_
- `功能相关 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/feature_related.html>`_

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

   guide/pre_training
   guide/supervised_fine_tuning
   guide/inference
   guide/deployment

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能特性
   :hidden:

   feature/start_tasks
   feature/ckpt
   feature/safetensors
   feature/configuration
   feature/logging
   feature/training_function
   feature/infer_function

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高阶开发
   :hidden:

   advanced_development/precision_optimization
   advanced_development/performance_optimization
   advanced_development/dev_migration
   advanced_development/multi_modal_dev
   advanced_development/accuracy_comparison
   advanced_development/api

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 优秀实践
   :hidden:

   example/distilled/distilled

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
