.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Federated 文档
=========================

联邦学习是一种加密的分布式机器学习技术，用于解决数据孤岛问题，在多方或者多资源计算节点间进行高效率，安全且可靠的机器学习。支持机器学习的各参与方在不直接共享本地数据的前提下，共建AI模型，包括但不限于广告推荐、分类、检测等主流深度学习模型，主要应用在金融，医疗，推荐等领域。

MindSpore Federated是一款开源联邦学习框架，提供样本联合的横向联邦模式和特征联合的纵向联邦模式。可支持面向亿级无状态终端设备的商用化部署，也可支持跨可信区的数据中心之间的云云联邦，可在用户数据不出本地的前提下，使能全场景智能应用。

使用MindSpore Federated横向框架的优势
----------------------------------------

横向联邦架构：

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_zh_cn/images/HFL.png" width="700px" alt="" >

1. 隐私安全

   支持基于多方安全计算（MPC）的精度无损的安全聚合方案，防止模型窃取。

   支持基于本地差分隐私的性能无损的加密方案，防止模型泄漏隐私数据。

   支持基于符号维度选择（SignDS）的梯度保护方案，防止模型隐私数据泄露的同时，可将通信开销降低99%。

2. 分布式联邦聚合

   云侧松耦合集群化处理方式，和分布式梯度二次聚合范式，支持千万级数量的大规模异构终端部署场景，实现高性能、高可用的联邦聚合计算，可应对网络不稳定，负载突变等问题。

3. 联邦效率提升

   支持自适应调频策略，支持梯度压缩算法，提高联邦学习效率，节省带宽资源。

   支持多种联邦聚合策略，提高联邦收敛的平滑度，兼顾全局和局部的精度最优化。

4. 灵活易用

   仅一行代码即可切换单机训练与联邦学习模式。

   网络模型可编程，聚合算法可编程，安全算法可编程，安全等级可定制。

   支持联邦训练模型的效果评估，提供联邦任务的监控能力。

使用MindSpore Federated纵向框架的优势
----------------------------------------

纵向联邦架构：

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_zh_cn/images/VFL.png" width="700px" alt="" >

1. 隐私安全

   支持高性能隐私集合求交协议（PSI），可防止联邦参与方获得交集外的ID信息，可应对数据不均衡场景。

   支持基于差分隐私的Label的加密方案，防止泄漏用户标签数据。

   支持软硬结合的隐私保护方案，和同态加密方式相比，可降低通信轮数和通信量。

2. 联邦训练

   支持多类型的拆分学习网络结构。

   面向大模型跨域训练，流水线并行优化。

使用MindSpore Federated的工作流程
-----------------------------------

1. `识别场景、积累数据 <https://www.mindspore.cn/federated/docs/zh-CN/master/image_classification_application.html#准备工作>`_

   识别出可使用联邦学习的业务场景，在客户端为联邦任务积累本地数据。

2. `模型选型、框架部署 <https://www.mindspore.cn/federated/docs/zh-CN/master/image_classification_application.html##生成端侧模型文件>`_

   进行模型原型的选型或开发，并使用工具生成方便部署的联邦学习模型。

3. `应用部署 <https://www.mindspore.cn/federated/docs/zh-CN/master/image_classification_application.html#模拟启动多客户端参与联邦学习>`_

   将对应组件部署到业务应用中，并在服务器上设置联邦配置任务和部署脚本。

常见应用场景
-----------------

1. `图像分类 <https://www.mindspore.cn/federated/docs/zh-CN/master/image_classification_application.html>`_

   使用联邦学习实现图像分类应用。

2. `文本分类 <https://www.mindspore.cn/federated/docs/zh-CN/master/sentiment_classification_application.html>`_

   使用联邦学习实现文本分类应用。

.. toctree::
   :maxdepth: 1
   :caption: 安装部署

   federated_install
   deploy_federated_server
   deploy_federated_client

.. toctree::
   :maxdepth: 1
   :caption: 横向应用实践

   image_classfication_dataset_process
   image_classification_application
   sentiment_classification_application
   image_classification_application_in_cross_silo
   object_detection_application_in_cross_silo

.. toctree::
   :maxdepth: 1
   :caption: 纵向应用实践

   data_join
   split_wnd_application
   split_pangu_alpha_application

.. toctree::
   :maxdepth: 1
   :caption: 安全和隐私

   local_differential_privacy_training_noise
   local_differential_privacy_training_signds
   pairwise_encryption_training
   private_set_intersection
   secure_vertical_federated_learning_with_TEE
   secure_vertical_federated_learning_with_DP

.. toctree::
   :maxdepth: 1
   :caption: 通信压缩

   communication_compression

.. toctree::
   :maxdepth: 1
   :caption: API参考

   Horizontal_FL_Server
   Horizontal_FL_Client
   Vertical_FL_Server
   Data_Join

.. toctree::
   :maxdepth: 1
   :caption: 参考文档

   faq
