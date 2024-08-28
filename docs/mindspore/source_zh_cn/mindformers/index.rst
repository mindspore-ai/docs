MindSpore Transformers 文档
=========================================

MindSpore Transformers（也称MindFormers）套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性，期望帮助用户轻松地实现大模型训练和创新研发。

如果您对MindFormers有任何建议，请通过 `issue <https://gitee.com/mindspore/mindformers/issues>`_ 与我们联系，我们将及时处理。

MindFormers支持一键启动任意任务的单卡/多卡训练、微调、评估、推理流程，它通过简化操作、提供灵活性和自动化流程，使得深度学习任务的执行变得更加高效和用户友好，用户可以通过以下说明文档进行学习：

- `开发迁移 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/dev_migration.html>`_
- `预训练 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/pre_training.html>`_
- `SFT微调 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/sft_tuning.html>`_
- `LoRA低参微调 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/lora_tuning.html>`_
- `评测 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/evaluation.html>`_
- `推理 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/inference.html>`_
- `量化 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/quantization.html>`_
- `MindIE服务器化部署 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/usage/mindie_deployment.html>`_

使用MindFormers进行灵活易用的个性化配置
--------------------------------------------

MindFormers以其强大的功能集，为用户提供了灵活且易于使用的个性化配置选项。具体来说，它具备以下几个关键特性：

1. `权重切分与合并 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/function/weight_conversion.html>`_

   不同分布式场景下的权重灵活地进行切分与合并。

2. `分布式并行 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/function/distributed_parallel.html>`_

   一键配置多维混合分布式并行，让模型在上至万卡的集群中高效运行。

3. `数据集 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/function/dataset.html>`_

   支持多种形式的数据集。

4. `断点续训 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/function/res_training.html>`_

   支持step级断点续训，有效减少大规模训练时意外中断造成的时间和资源浪费。

使用MindFormers进行深度调优
------------------------------------

- `精度调优 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/acc_optimize/acc_optimize.html>`_
- `性能调优 <https://www.mindspore.cn/docs/zh-CN/master/mindformers/perf_optimize/perf_optimize.html>`_

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
   usage/lora_tuning
   usage/evaluation
   usage/inference
   usage/mindie_deployment
   usage/quantization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能说明
   :hidden:

   function/weight_conversion
   function/distributed_parallel
   function/dataset
   function/res_training

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 精度调优
   :hidden:

   acc_optimize/acc_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 性能调优
   :hidden:

   perf_optimize/perf_optimize

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
   :caption: FAQ
   :hidden:

   faq/model_related
   faq/func_related
   faq/mindformers_contribution
   faq/openmind_contribution
