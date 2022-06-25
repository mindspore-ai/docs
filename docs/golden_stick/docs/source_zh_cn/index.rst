MindSpore Golden Stick 文档
=============================

MindSpore Golden Stick是一个开源的模型压缩算法集，提供了一套应用算法的用户接口，让用户能够统一方便地使用例如量化、剪枝等等模型压缩算法。金箍棒同时为算法开发者提供修改网络定义的基础能力，在算法和网络定义中间抽象了一层IR，对算法开发者屏蔽具体的网络定义，使其能聚焦于算法逻辑的开发上。后面简称MindSpore Golden Stick为金箍棒。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_zh_cn/images/golden-stick-arch.png" width="700px" alt="" >

设计思路
--------

1. 提供统一的算法API，降低用户应用算法的学习成本

   模型压缩算法种类繁多，有如量化感知训练算法、剪枝算法、矩阵分解算法、知识蒸馏算法等；在每类压缩算法中，还有会各种具体的算法，比如LSQ、PACT都是量化感知训练算法。不同算法的应用方式往往各不相同，这增加了用户应用算法的学习成本。金箍棒对算法应用流程做了梳理和抽象，提供了一套统一的算法应用接口，最大程度缩减算法应用的学习成本。同时这也方便了后续在算法生态的基础上，做一些组合算法或算法搜优的探索。

2. 提供修改网络定义的能力，降低算法接入成本

   模型压缩算法往往会针对特定的网络结构做设计或者优化，而很少关注具体的网络定义。金箍棒提供了通过接口修改前端网络定义的能力，让算法开发者聚焦于算法的实现，而不用对不同的网络定义重复造轮子。此外金箍棒还会提供了一些调测能力，包括网络dump、逐层profiling、算法效果分析、可视化等能力，旨在帮助算法开发者提升开发和研究效率，帮助用户寻找契合于自己需求的算法。

规划
---------

- 金箍棒初始版本包含一个稳定的API，并提供一个线性量化算法，一个非线性量化算法和一个结构化剪枝算法。后续会提供更多的算法和更完善的网络支持，调测能力也会在后续版本提供。将来随着算法的丰富，金箍棒还会探索算法搜优、算法硬件自适应和NAS等能力，敬请期待。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 量化算法

   quantization/quantization
   quantization/qbnn

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
