MindSpore Golden Stick 文档
=============================

MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集。MindSpore Golden Stick的架构图如下图所示，分为五个部分：

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_zh_cn/images/golden-stick-arch.png" width="700px" alt="" >

1. 底层的MindSpore Rewrite模块提供修改前端网络的能力，基于此模块提供的接口，算法开发者可以按照特定的规则对MindSpore的前端网络做节点和拓扑关系的增删查改；

2. 基于MindSpore Rewrite这个基础能力，MindSpore Golden Stick会提供各种类型的算法，比如SimQAT算法、SLB量化算法、SCOP剪枝算法等；

3. 在算法的更上层，MindSpore Golden Stick还规划了如AMC（自动模型压缩技术）、NAS（网络结构搜索）、HAQ（硬件感知的自动量化）等高阶技术；

4. 为了方便开发者分析调试算法，MindSpore Golden Stick提供了一些工具，如Visualization工具（可视化工具）、Profiler工具（逐层分析工具）、Summary工具（算法压缩效果分析工具）等；

5. 在最外层，MindSpore Golden Stick封装了一套简洁的用户接口。

.. note:
 架构图是MindSpore Golden Stick的全貌，其中包含了当前版本已经实现的功能以及规划在RoadMap中能力。具体开放的功能可以参考对应版本的ReleaseNotes。

设计思路
---------------------------------------

MindSpore Golden Stick除了提供丰富的模型压缩算法外，一个重要的设计理念是针对业界种类繁多的模型压缩算法，提供给用户一个尽可能统一且简洁的体验，降低用户的算法应用成本。MindSpore Golden Stick通过两个举措来实现该理念：

1. 统一的算法接口设计，降低用户应用成本

   模型压缩算法种类繁多，有如量化感知训练算法、剪枝算法、矩阵分解算法、知识蒸馏算法等；在每类压缩算法中，还有会各种具体的算法，比如LSQ、PACT都是量化感知训练算法。不同算法的应用方式往往各不相同，这增加了用户应用算法的学习成本。MindSpore Golden Stick对算法应用流程做了梳理和抽象，提供了一套统一的算法应用接口，最大程度缩减算法应用的学习成本。同时这也方便了后续在算法生态的基础上，做一些AMC（自动模型压缩技术）、NAS（网络结构搜索）、HAQ（硬件感知的自动量化）等高阶技术的探索。

2. 提供前端网络修改能力，降低算法接入成本

   模型压缩算法往往会针对特定的网络结构做设计或者优化，如感知量化算法往往在网络中的Conv2d、Conv2d + BatchNorm2d或者Conv2d + BatchNorm2d + Relu结构上插入伪量化节点。MindSpore Golden Stick提供了通过接口修改前端网络的能力，算法开发者可以基于此能力制定通用的改图规则去实现算法逻辑，而不需要对每个特定的网络都实现一遍算法逻辑算法。此外MindSpore Golden Stick还会提供一些调测能力，包括网络dump、逐层profiling、算法效果分析、可视化等能力，旨在帮助算法开发者提升开发和研究效率，帮助用户寻找契合于自己需求的算法。

应用MindSpore Golden Stick算法的一般流程
-----------------------------------------

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_zh_cn/images/workflow.png" width="800px" alt="" >

1. 训练阶段

在训练网络时应用MindSpore Golden Stick算法不会对原有的训练脚本逻辑产生很大的影响，如上图中黄色部分所示，仅需要增加额外两步：

- **应用MindSpore Golden Stick算法优化网络：** 在原训练流程中，在定义原始网络之后，网络训练之前，应用MindSpore Golden Stick算法优化网络结构。一般这个步骤是调用MindSpore Golden Stick的`apply`接口实现的，可以参考 `应用SimQAT算法 <https://mindspore.cn/golden_stick/docs/zh-CN/master/quantization/simqat.html#%E5%BA%94%E7%94%A8%E9%87%8F%E5%8C%96%E7%AE%97%E6%B3%95>`_。

- **注册MindSpore Golden Stick回调逻辑：** 将MindSpore Golden Stick算法的回调逻辑注册到要训练的model中。一般这个步骤是调用MindSpore Golden Stick的`callback`获取相应的callback对象， `注册到model <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/callback.html>`_ 中。

2. 部署阶段

- **网络转换：** 经过MindSpore Golden Stick压缩的网络可能需要额外的步骤，将网络中模型压缩相关的结构从训练形态转化为部署形态，方便进一步进行模型导出和模型部署。比如在对于感知量化场景，常常需要将网络中的伪量化节点消除，转换为网络中的算子属性。当前版本未开放该能力。

.. note::
 - 应用MindSpore Golden Stick算法的细节，可以在每个算法章节中找到详细说明和示例代码。
 - 流程中的"网络训练或重训"步骤可以参考 `MindSpore训练与评估 <https://mindspore.cn/tutorials/zh-CN/master/advanced/train/train_eval.html>`_ 章节。
 - 流程中的"ms.export"步骤可以参考 `导出mindir格式文件 <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/save.html#%E5%AF%BC%E5%87%BAmindir%E6%A0%BC%E5%BC%8F%E6%96%87%E4%BB%B6>`_ 章节。
 - 流程中的"昇思推理优化工具和运行时"步骤可以参考 `昇思推理 <https://mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html>`_ 章节。

未来规划
----------

MindSpore Golden Stick初始版本包含一个稳定的API，并提供一个线性量化算法，一个非线性量化算法和一个结构化剪枝算法。后续会提供更多的算法和更完善的网络支持，调测能力也会在后续版本提供。将来随着算法的丰富，MindSpore Golden Stick还会探索AMC、HAQ和NAS等能力，敬请期待。

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
