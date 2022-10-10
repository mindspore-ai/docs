MindScience 文档
===================

MindScience是基于昇思MindSpore融合架构打造的科学计算行业套件，包含了业界领先的数据集、基础模型、预置高精度模型和前后处理工具，加速了科学行业应用开发。目前已推出面向电子信息行业的MindElec套件和面向生命科学行业的MindSPONGE套件，分别实现了电磁仿真性能提升10倍和生物制药化合物模拟效率提升50%。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindscience/docs/source_zh_cn/mindelec/images/mindscience_cn.png" width="700px" alt="" >

使用MindScience的典型场景
------------------------------

1. `电磁仿真 <https://www.mindspore.cn/mindscience/docs/zh-CN/master/mindelec/point_cloud.html>`_

   高效、易用的AI电磁仿真套件，提供了多模数据转换、高维数据编码以及创新的AI电磁仿真模型等功能，已在手机容差等场景中实现性能提升10倍、仿真精度媲美传统科学计算的技术突破。

2. `计算生物 <./mindsponge/why.html>`_

   MindSpore SPONGE(Simulation Package tOwards Next GEneration molecular modelling)是基于昇思MindSpore的计算生物领域套件，支持分子动力学、蛋白质折叠等常用功能，旨在于为广大的科研人员、老师及学生提供高效易用的AI计算生物软件。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: MindSPONGE

   mindsponge/why
   mindsponge/intro
   mindsponge/xponge
   mindsponge/sponge
   mindsponge/cybertron
   mindsponge/MEGAProtein
   mindsponge

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: MindElec

   mindelec/intro_and_install
   mindelec/physics_driven
   mindelec/data_driven
   mindelec/visualization
   mindelec
