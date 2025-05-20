MindSpore Transformers 文档
=========================================

MindSpore Transformers（也称MindFormers）套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性，期望帮助用户轻松地实现大模型训练和创新研发。

用户可以参阅 `整体架构 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/overview.html>`_ 和 `模型库 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/models.html>`_ 来初步了解MindFormers的架构和模型支持度；参考 `安装 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/quick_start/install.html>`_ 和 `快速启动 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/quick_start/source_code_start.html>`_ 章节，迅速上手MindFormers。

如果您对MindFormers有任何建议，请通过 `issue <https://gitee.com/mindspore/mindformers/issues>`_ 与我们联系，我们将及时处理。

MindFormers支持一键启动任意任务的单卡/多卡训练、微调、评估、推理流程，它通过简化操作、提供灵活性和自动化流程，使得深度学习任务的执行变得更加高效和用户友好，用户可以通过以下说明文档进行学习：

- `开发迁移 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/dev_migration.html>`_
- `预训练 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/pre_training.html>`_
- `SFT微调 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/sft_tuning.html>`_
- `评测 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/evaluation.html>`_
- `推理 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/inference.html>`_
- `量化 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/quantization.html>`_
- `服务化部署 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/mindie_deployment.html>`_
- `动态图并行开发 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/pretrain_gpt.html>`_
- `多模态理解模型开发指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/multi_modal.html>`_

使用MindFormers进行灵活易用的个性化配置
--------------------------------------------

MindFormers以其强大的功能集，为用户提供了灵活且易于使用的个性化配置选项。具体来说，它具备以下几个关键特性：

1. `权重格式转换 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html>`_

   提供统一的权重转换工具，能够将模型权重在HuggingFace所使用的格式与MindFormers所使用的格式之间相互转换。

2. `分布式权重切分与合并 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html>`_

   不同分布式场景下的权重灵活地进行切分与合并。

3. `分布式并行 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/distributed_parallel.html>`_

   一键配置多维混合分布式并行，让模型在上至万卡的集群中高效运行。

4. `数据集 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/dataset.html>`_

   支持多种形式的数据集。

5. `权重保存与断点续训 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/resume_training.html>`_

   支持step级断点续训，有效减少大规模训练时意外中断造成的时间和资源浪费。

6. `训练指标监控 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/monitor.html>`_

   提供大模型训练阶段的可视化服务，用于监控和分析训练过程中的各种指标和信息。

7. `训练高可用 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/high_availability.html>`_

   提供大模型训练阶段的高可用能力，包括临终 CKPT 保存、UCE 故障容错恢复和进程级重调度恢复功能。

8. `Safetensors权重 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/safetensors.html>`_

   支持safetensors格式的权重文件保存及加载功能。

使用MindFormers进行深度调优
------------------------------------

- `精度调优 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/acc_optimize/acc_optimize.html>`_
- `性能调优 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/perf_optimize/perf_optimize.html>`_

附录
------------------------------------

- `环境变量说明 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/env_variables.html>`_
- `配置文件说明 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html>`_

FAQ
------------------------------------

- `模型相关 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/model_related.html>`_
- `功能相关 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/func_related.html>`_
- `MindFormers贡献指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/mindformers_contribution.html>`_
- `魔乐社区贡献指南 <https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/modelers_contribution.html>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 开始
   :hidden:

   start/overview
   start/models

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速入门
   :hidden:

   quick_start/install
   quick_start/source_code_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用教程
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
   :caption: 功能说明
   :hidden:

   function/weight_conversion
   function/transform_weight
   function/distributed_parallel
   function/dataset
   function/resume_training
   function/monitor
   function/high_availability
   function/safetensors

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 精度调优
   :hidden:

   acc_optimize/acc_optimize
   acc_optimize/pynative_acc_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 性能调优
   :hidden:

   perf_optimize/perf_optimize

.. toctree::
   :maxdepth: 1
   :caption: API参考
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
   :caption: 附录
   :hidden:

   appendix/env_variables
   appendix/conf_files

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 案例
   :hidden:

   examples/distilled

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ
   :hidden:

   faq/model_related
   faq/func_related
   faq/mindformers_contribution
   faq/modelers_contribution
