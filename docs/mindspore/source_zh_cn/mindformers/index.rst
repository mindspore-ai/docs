大模型开发
====================================

MindSpore Transformers（也称MindFormers）套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性，期望帮助用户轻松地实现大模型训练和创新研发。

如果您对MindFormers有任何建议，请通过`issue <https://gitee.com/mindspore/mindformers/issues>`_ 与我们联系，我们将及时处理。

MindFormers支持一键启动任意任务的单卡/多卡训练、微调、评估、推理流程，它通过简化操作、提供灵活性和自动化流程，使得深度学习任务的执行变得更加高效和用户友好，用户可以通过以下说明文档进行使用和学习：

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 开始

   start/overview
   start/models

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 快速入门

   quick_start/install
   quick_start/source_code_start

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用教程

   usage/dev_migration
   usage/pre_training
   usage/sft_tuning
   usage/lora_tuning
   usage/evaluation
   usage/inference
   usage/mindie_deployment
   usage/quantization

使用MindFormers进行灵活易用的个性化配置
--------------------------------------------

MindFormers以其强大的功能集，为用户提供了灵活且易于使用的个性化配置选项。具体来说，它具备以下几个关键特性：

- 权重转换功能：允许不同框架训练的模型权重在MindFormers框架中使用，提高了模型的兼容性和可移植性。

- 分布式并行：通过在多个NPU上并行化计算，可以加速模型的训练过程。

- 在线加载数据集：允许在线加载预训练的模型，无需下载和安装大型文件，简化了使用流程，节省了宝贵的存储资源。

- 断点续训功能：MindFormers的断点续训功能确保即使在训练过程中遇到中断，也能够最大限度地保留和利用已经完成的训练工作。

具体功能使用说明如下：

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 功能说明

   function/weight_conversion
   function/distributed_parallel
   function/dataset
   function/res_training

使用MindFormers进行精度和性能调优
------------------------------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 精度调优

   acc_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 性能调优

   perf_optimize/perf_optimize

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 附录

   appendix/env_variables
   appendix/conf_files

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: FAQ

   faq/model_related
   faq/func_related
   faq/mindformers_contribution
   faq/openmind_contribution
