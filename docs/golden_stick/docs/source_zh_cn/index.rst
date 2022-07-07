MindSpore Golden Stick 文档
=============================

MindSpore Golden Stick是一个开源的模型压缩算法集，提供了一套应用算法的用户接口，让用户能够统一方便地使用例如量化、剪枝等模型压缩算法。金箍棒同时为算法开发者提供修改网络定义的基础能力，在算法和网络定义中间抽象了一层IR，对算法开发者屏蔽具体的网络定义，使其能聚焦于算法逻辑的开发上。后面简称MindSpore Golden Stick为金箍棒。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/docs/golden_stick/docs/source_zh_cn/images/golden-stick-arch.png" width="700px" alt="" >

设计思路
---------------------------------------

1. 提供统一的算法API，降低用户应用算法的学习成本

   模型压缩算法种类繁多，有如量化感知训练算法、剪枝算法、矩阵分解算法、知识蒸馏算法等；在每类压缩算法中，还有会各种具体的算法，比如LSQ、PACT都是量化感知训练算法。不同算法的应用方式往往各不相同，这增加了用户应用算法的学习成本。金箍棒对算法应用流程做了梳理和抽象，提供了一套统一的算法应用接口，最大程度缩减算法应用的学习成本。同时这也方便了后续在算法生态的基础上，做一些AMC（自动模型压缩技术）、NAS（网络结构搜索）等技术的探索。

2. 提供修改网络定义的能力，降低算法接入成本

   模型压缩算法往往会针对特定的网络结构做设计或者优化，而很少关注具体的网络定义。金箍棒提供了通过接口修改前端网络定义的能力，让算法开发者聚焦于算法的实现，而不用对不同的网络定义重复造轮子。此外金箍棒还会提供了一些调测能力，包括网络dump、逐层profiling、算法效果分析、可视化等能力，旨在帮助算法开发者提升开发和研究效率，帮助用户寻找契合于自己需求的算法。

应用金箍棒算法的一般流程
---------------------------------------

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/docs/golden_stick/docs/source_zh_cn/images/workflow.png" width="800px" alt="" >

1. 训练阶段

在训练网络时应用金箍棒算法不会对原有的训练脚本逻辑产生很大的影响，如上图中黄色部分所示，仅需要增加额外两步：

- **应用金箍棒算法优化模型：** 在原训练流程中，在定义原始网络之后，模型训练之前，应用金箍棒算法优化网络结构。一般这个步骤是调用金箍棒的`apply`接口实现的，可以参考[应用SimQAT算法](https://mindspore.cn/golden_stick/docs/zh-CN/r1.8/quantization/simqat.html#%E5%BA%94%E7%94%A8%E9%87%8F%E5%8C%96%E7%AE%97%E6%B3%95)。

- **注册金箍棒回调接口：** 将金箍棒算法的回调算法注册到要训练的Model中。一般这个步骤是调用金箍棒的`callback`获取相应的callback对象，并[注册到Model](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/callback.html)中。

2. 部署阶段

- **网络转换：** 经过金箍棒压缩的网络可能需要额外的步骤，将网络中模型压缩相关的结构从训练形态转化为部署形态，方便进一步进行模型导出和模型部署。比如在对于感知量化场景，常常需要将网络中的伪量化节点消除，转换为网络中的算子属性。

.. note::
 - 应用金箍棒算法的细节，可以在每个算法章节中找到详细说明和示例代码。
 - 流程中的"网络训练或重训"步骤可以参考[MindSpore训练与评估](https://mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html)章节。
 - 流程中的"ms.export"步骤可以参考[导出mindir格式文件](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/save.html#%E5%AF%BC%E5%87%BAmindir%E6%A0%BC%E5%BC%8F%E6%96%87%E4%BB%B6)章节。
 - 流程中的"模型优化"和"模型导出"步骤可以参考[昇思推理离线工具](https://mindspore.cn/lite/docs/zh-CN/r1.8/use/converter_tool.html)章节。
 - 流程中的"昇思推理运行时"步骤可以参考[昇思推理运行时](https://mindspore.cn/lite/docs/zh-CN/r1.8/use/runtime.html)章节。

规划
---------------------------------------

金箍棒初始版本包含一个稳定的API，并提供一个线性量化算法，一个非线性量化算法和一个结构化剪枝算法。后续会提供更多的算法和更完善的网络支持，调测能力也会在后续版本提供。将来随着算法的丰富，金箍棒还会探索自动模型压缩（AMC）、算法硬件自适应（HAQ）和网络结构搜索（NAS）等能力，敬请期待。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 量化算法

   quantization/overview
   quantization/simqat
   quantization/slb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 剪枝算法

   pruner/overview
   pruner/scop

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindspore_gs
