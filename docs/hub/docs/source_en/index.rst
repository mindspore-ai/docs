MindSpore Hub Documents
=========================

MindSpore Hub is a pre-trained model application tool of the MindSpore ecosystem. It provides the following functions:

- Plug-and-play model loading
- Easy-to-use transfer learning

.. code-block::

   import mindspore
   import mindspore_hub as mshub
   from mindspore import set_context, GRAPH_MODE
   
   set_context(mode=GRAPH_MODE,
                       device_target="Ascend",
                       device_id=0)
   
   model = "mindspore/1.6/googlenet_cifar10"
   
   # Initialize the number of classes based on the pre-trained model.
   network = mshub.load(model, num_classes=10)
   network.set_train(False)
   
   # ...

Code repository address: <https://gitee.com/mindspore/hub>

Typical Application Scenarios
--------------------------------------------

1. `Inference Validation <https://www.mindspore.cn/hub/docs/en/master/loading_model_from_hub.html#for-inference-validation>`_

   With only one line of code, use mindspore_hub.load to load the pre-trained model.

2. `Transfer Learning <https://www.mindspore.cn/hub/docs/en/master/loading_model_from_hub.html#for-transfer-training>`_

   After loading models using mindspore_hub.load, add an extra argument to load the feature extraction of the neural network. This makes it easier to add new layers for transfer learning.

3. `Model Releasing <https://www.mindspore.cn/hub/docs/en/master/publish_model.html>`_

   Release the trained model to MindSpore Hub according to the specified procedure for download and use.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   hub_installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   loading_model_from_hub
   publish_model

.. toctree::
   :maxdepth: 1
   :caption: API References

   hub

.. toctree::
   :maxdepth: 1
   :caption: Models

   MindSpore Hub↗ <https://www.mindspore.cn/hub>
