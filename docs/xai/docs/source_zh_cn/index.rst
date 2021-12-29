MindSpore XAI 文档
===========================

MindSpore XAI是一个基于昇思MindSpore的可解释AI工具箱。当前深度学习模型多为黑盒模型，性能表现好但可解释性较差。XAI旨在为用户提供对模型决策的解释，帮助用户更好地理解模型、信任模型，以及当模型出现错误时有针对性地改进模型。除了提供多种解释方法，还提供了一套对解释方法效果评分的度量方法，从多种维度评估解释方法的效果，从而帮助用户比较和选择最适合于特定场景的解释方法。

.. image:: ./images/xai_cn.png
  :width: 700px

使用MindSpore XAI的典型场景
---------------------------

1. `使用解释器 <https://www.mindspore.cn/xai/docs/zh-CN/master/using_explainers.html>`_

   为图片分类模型输出热力图解释。

2. `使用度量方法 <https://www.mindspore.cn/xai/docs/zh-CN/master/using_benchmarks.html>`_

   为解释器作出优劣评估。

3. `使用MindInsight <https://www.mindspore.cn/xai/docs/zh-CN/master/using_mindinsight.html>`_

   可视化解释器及度量方法输出的结果。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   installation


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   using_explainers
   using_benchmarks
   using_mindinsight
