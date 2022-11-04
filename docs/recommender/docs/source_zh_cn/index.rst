MindSpore Recommender 文档
=============================

MindSpore Recommender是一个构建在MindSpore框架基础上，面向推荐领域的开源训练加速库，通过MindSpore大规模的异构计算加速能力，MindSpore Recommender支持在线以及离线场景大规模动态特征的高效训练。

.. raw:: html

   <p style="text-align: center;"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/recommender/docs/source_zh_cn/images/architecture.png" width="600px" alt="" ></p>

MindSpore Recommender加速库由如下部分组成：

- 在线训练：通过流式读取实时数据源中的数据 (例如：Kafka)，以及在线的实时数据加工，实现实时数据的在线训练以及增量模型更新，从而支持对于模型有实时更新需要的业务场景；
- 离线训练：面向传统的离线数据集训练场景，通过自动并行、分布式特征缓存、异构加速等技术方案，支持包含大规模特征向量的推荐模型训练；
- 数据处理：MindPandas和MindData提供了在离线数据的读取和处理能力，通过全Python的表达支持，节省了多语言和多框架开销，同时打通了数据处理和模型训练的高效数据流转链路；
- 模型库：包含持续丰富的典型推荐模型训练，经过严格的精度和性能验证，支持开箱即用。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   offline_learning
   online_learning

.. toctree::
   :maxdepth: 1
   :caption: API参考

   recommender
