MindSpore XAI Documents
===========================

Currently, most deep learning models are black-box models with good performance but poor explainability. The MindSpore XAI - a MindSpore-based explainable AI toolbox - provides a variety of explanation and decision methods to help you better understand, trust, and improve models. It also evaluates the explanation methods from various dimensions, enabling you to compare and select methods best suited to your environment.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/xai/docs/source_en/images/xai_en.png" width="700px" alt="" >

Typical Application Scenarios
------------------------------

1. `Using Explainers <https://www.mindspore.cn/xai/docs/en/master/using_explainers.html>`_

   Explain image classification models using saliency maps.

2. `Benchmarks <https://www.mindspore.cn/xai/docs/en/master/using_benchmarks.html>`_

   Scores explainers.

3. `MindInsight <https://www.mindspore.cn/xai/docs/en/master/using_mindinsight.html>`_

   Visualize the results from Explainers and Benchmarks.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   installation


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   using_explainers
   using_benchmarks
   using_mindinsight

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindspore_xai.runner
   mindspore_xai.explanation
   mindspore_xai.benchmark
