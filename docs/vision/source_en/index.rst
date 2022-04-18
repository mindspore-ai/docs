MindSpore Vision Documents
==========================

MindSpore Vision is an open source computer vision research toolboxbased on MindSpore framework, which includes classification, detection (to do), video (to do) and 3D (to do). MindSpore Vision is designed to serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods. We also hope to invite you to participate in our development of MindSpore Vision.

Scenarios where MindSpore Vision is applied
----------------------------

- `Classification <https://gitee.com/mindspore/vision/blob/master/mindvision/classification/README_en.md>`_
    Image classification toolbox and benchmark.
- `Detection <https://gitee.com/mindspore/vision/blob/master/mindvision/detection/README.md>`_
    Image detection toolbox and benchmark.(to do)

Base Structure
--------------

MindSpore Vision is a MindSpore-based Python package that provides high-level features:

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/vision/source_en/images/BaseArchitecture.png" width="700px" alt="" >

- Classification: Deep neural networks work flows built on an engine system.
- Backbone: Base backbone of models like ResNet and MobileNet series.
- Engine: Callback functions for model training.
- Dataset: Domain oriented rich dataset interface.
- Utils: Rich visualization and IO(Input/Output) interfaces.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   mindvision_install

.. toctree::
   :maxdepth: 1
   :caption: API References

   classification
   engine
   utils
   