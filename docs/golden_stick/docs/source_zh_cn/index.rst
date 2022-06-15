MindSpore Golden Stick 文档
=============================

MindSpore Golden Stick是一个开源的模型压缩算法集，并提供了一套应用算法的用户接口，让用户能够统一方便地使用例如量化、剪枝等等模型压缩算法。
MindSpore Golden Stick同时为算法开发者提供修改网络定义的基础设施，在算法和网络定义中间抽象了一层IR，对算法开发者屏蔽具体的网络定义，使其能聚焦与算法逻辑的开发上。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/golden_stick/docs/source_zh_cn/images/golden-stick-arch.png" width="700px" alt="" >

设计思路
--------

1. 提供以用户为中心的API，降低用户学习成本

   MindSpore Golden Stick定义了一个算法的抽象类，所有的算法实现都应该继承于此基类，而用户也可以使用基类定义的接口直接应用所有的算法，而不需要针对每一个算法都学习其应用方式。这也方便了后续在算法生态的基础上，做一些组合算法或算法搜优的探索。

2. 提供一些基础设施能力，降低算法接入成本

   MindSpore Golden Stick提供了用于算法实现一些基础设施能力，比如调测、网络修改等能力。调测能力主要包括网络dump、逐层profiling等能力，旨在帮助算法开发者定位算法实现中的bug，帮助用户寻找契合于自己需求的算法。网络修改能力是指通过一系列API，修改Python定义的网络结构的能力，旨在让算法开发者聚焦于算法的实现，而不用对不同的网络定义重复造轮子。

规划
---------

- MindSpore Golden Stick初始版本包含一个稳定的API，并提供两个线性量化算法，一个非线性量化算法和一个结构化剪枝算法。后续会提供更多的算法和更完善的网络支持、调测能力等。后期随着算法的丰富，还规划了组合算法和算法搜优的能力，敬请期待。

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

   pruner/pruner
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