MindSpore Hub Documents
=========================

MindSpore Hub provides pre-trained model applications in the MindSpore ecosystem.

MindSpore Hub provides the following functions:

- Plug-and-play model loading
- Easy-to-use transfer learning

.. code-block::

   import mindspore
   import mindspore_hub as mshub
   from mindspore import context
   
   context.set_context(mode=context.GRAPH_MODE,
                       device_target="Ascend",
                       device_id=0)
   
   model = "mindspore/ascend/0.7/googlenet_v1_cifar10"
   
   # Initialize the number of classes based on the pre-trained model.
   network = mshub.load(model, num_classes=10)
   network.set_train(False)
   
   # ...

Typical MindSpore Hub Application Scenarios
--------------------------------------------

1. `Inference Validation <https://www.mindspore.cn/hub/docs/en/master/loading_model_from_hub.html#for-inference-validation>`_

   Use mindspore_hub.load to load the pre-trained model with only a line of code.

2. `Transfer Learning <https://www.mindspore.cn/hub/docs/en/master/loading_model_from_hub.html#for-transfer-training>`_

   After the model is loaded by using mindspore_hub.load, add an extra parameter to load only the feature extraction part of the neural network. In this way, some new layers can be easily added for transfer learning.

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


