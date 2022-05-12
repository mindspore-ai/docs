MindSpore XAI Documents
===========================

Currently, most deep learning models are black-box models with good performance but poor explainability. The MindSpore XAI - a MindSpore-based explainable AI toolbox - provides a variety of explanation and decision methods to help you better understand, trust, and improve models. It also evaluates the explanation methods from various dimensions, enabling you to compare and select methods best suited to your environment.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/xai/docs/source_en/images/xai_en.png" width="700px" alt="" >

Typical Application Scenarios
------------------------------

1. `Using Explainers <https://www.mindspore.cn/xai/docs/en/master/using_cv_explainers.html>`_

   Explain image classification models using saliency maps.

2. `Benchmarks <https://www.mindspore.cn/xai/docs/en/master/using_cv_benchmarks.html>`_

   Scores CV explainers.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   using_cv_explainers
   using_cv_benchmarks
   using_tabular_explainers
   using_tabsim
   using_tbnet

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindspore_xai.explainer
   mindspore_xai.benchmark
   mindspore_xai.tool
