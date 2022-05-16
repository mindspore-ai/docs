MindSpore XAI 文档
===========================

MindSpore XAI是一个基于昇思MindSpore的可解释AI工具箱。当前深度学习模型多为黑盒模型，性能表现好但可解释性较差。XAI旨在为用户提供对模型决策的解释，帮助用户更好地理解模型、信任模型，以及当模型出现错误时有针对性地改进模型。除了提供多种解释方法，还提供了一套对解释方法效果评分的度量方法，从多种维度评估解释方法的效果，从而帮助用户比较和选择最适合于特定场景的解释方法。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/xai/docs/source_zh_cn/images/xai_cn.png" width="700px" alt="" >

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   using_cv_explainers
   using_cv_benchmarks
   using_tabular_explainers
   using_tabsim
   using_tbnet

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindspore_xai.explainer
   mindspore_xai.benchmark
   mindspore_xai.tool
