MindSpore Probability文档
==========================

深度学习模型具有强大的拟合能力，而贝叶斯理论具有很好的可解释能力。昇思MindSpore概率编程提供了贝叶斯学习和深度学习“无缝”融合的框架，旨在为用户提供完善的概率学习库，用于建立概率模型和应用贝叶斯推理。

概率编程主要包括以下几部分：

- 提供丰富的统计分布和常用的概率推断算法。
- 提供可组合的概率编程模块，让开发者可以用开发深度学习模型的逻辑来构造深度概率模型。
- 提供不确定估计和异常检测的工具箱，拓展贝叶斯应用功能。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/probability/docs/source_zh_cn/probability_cn.png" width="700px" alt="" >

使用概率编程的典型场景
-----------------------

1. `构建贝叶斯神经网络 <https://www.mindspore.cn/probability/docs/zh-CN/master/using_bnn.html>`_
   
   利用贝叶斯神经网络实现图片分类应用。
   
2. `构建变分自编码器 <https://www.mindspore.cn/probability/docs/zh-CN/master/using_the_vae.html>`_
   
   利用变分自编码器压缩输入数据，生成新样本。
   
3. `DNN一键转BNN <https://www.mindspore.cn/probability/docs/zh-CN/master/one_click_conversion_from_dnn_to_bnn.html>`_
   
   支持DNN模型一键转换成BNN模型。
   
4. `使用不确定性估计工具箱 <https://www.mindspore.cn/probability/docs/zh-CN/master/using_the_uncertainty_toolbox.html>`_
   
   利用不确定性估计工具箱，得到偶然不确定性和认知不确定性，更好地理解模型和数据集。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   using_bnn
   using_the_vae
   one_click_conversion_from_dnn_to_bnn
   using_the_uncertainty_toolbox
   probability

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindspore.nn.probability
