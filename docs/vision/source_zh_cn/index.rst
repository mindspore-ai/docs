MindSpore Vision 文档
=========================

MindSpore Vision是一个开源的基于MindSpore框架的计算机视觉研究工具箱，工具所涉及的任务主要包括分类、检测(开发中)、视频(开发中)和3D(开发中)。MindSpore Vision旨在通过提供易用的接口来帮助您复现现有的经典模型并开发自己的新模型。同时也邀请您参与到我们的开发中。

使用MindSpore Vision的典型场景
----------------------------------

- `分类 <https://gitee.com/mindspore/vision/blob/master/mindvision/classification/README_en.md>`_
   图像分类工具箱和基准。

基本结构
-----------

MindSpore Vision是一个基于MindSpore的Python软件包。

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/vision/source_zh_cn/images/BaseArchitecture.png" width="700px" alt="" >

提供的高级功能包括：

- 分类: 建立在引擎系统上的深层神经网络工作流程。
- 主干网络: ResNet和MobileNet等模型的基础主干网络。
- 引擎: 用于模型训练的回调函数。
- 数据集: 面向各个领域的丰富的数据集接口。
- 工具: 丰富的可视化接口和IO(输入/输出)接口。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   mindvision_install

.. toctree::
   :maxdepth: 1
   :caption: API参考

   classification
   engine
   utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE