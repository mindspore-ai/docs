.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png
    :target: https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/source_en/advanced/modules.rst

Model Module Customization
===========================

.. toctree::
  :maxdepth: 1
  :hidden:
  
  modules/layer
  modules/initializer
  modules/loss
  modules/optimizer

Basic Usage Examples
--------------------

The neural network model is composed of various layers. MindSpore
provides Cell, the base unit for constructing neural network layers, and
performs neural network encapsulation based on Cell. In the following,
the classical model AlexNet is constructed by using Cell.

.. figure:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/modules/images/AlexNet.ppm
   :alt: alextnet

As shown in the figure, AlexNet consists of five convolutional layers in
series with three fully-connected layers. We construct it by using the
neural network layer interface provided by ``mindspore.nn``.

.. code:: 

    from mindspore import nn

The following code shows how to quickly construct AlexNet by using
``nn.Cell``.

-  Top-level neural networks inherit from ``nn.Cell`` as a nested
   structure.
-  Each neural network layer is a subclass of ``nn.Cell``.
-  ``nn.SequentialCell`` can be simplified when defining models for
   sequential structures.

.. code:: python

    class AlexNet(nn.Cell):
        def __init__(self, num_classes=1000, dropout=0.5):
            super().__init__()
            self.features = nn.SequentialCell(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, pad_mode='pad', padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, pad_mode='pad', padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, pad_mode='pad', padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, pad_mode='pad', padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.SequentialCell(
                nn.Dropout(1-dropout),
                nn.Dense(256 * 6 * 6, 4096),
                nn.ReLU(),
                nn.Dropout(1-dropout),
                nn.Dense(4096, 4096),
                nn.ReLU(),
                nn.Dense(4096, num_classes),
            )
    
        def construct(self, x):
            x = self.features(x)
            x = x.view(x.shape[0], 256 * 6 * 6)
            x = self.classifier(x)
            return x

In the process of defining a model, the ``construct`` method can be used within Python syntax for any construction of the model
structure, such as conditional, looping, and other control flow
statements. However, when compiling Just In Time, the syntax needs to
be parsed by the compiler. For a syntax restriction, refer to:
`Static diagram syntax
support <https://www.mindspore.cn/docs/en/r2.0.0-alpha/note/static_graph_syntax_support.html>`_ .

After completing the model construction, we construct a single sample of
data and send it to the instantiated AlexNet to find the positive
results.

.. code:: python

    import numpy as np
    import mindspore
    from mindspore import Tensor
    
    x = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)

.. code:: python

    network = AlexNet()
    logits = network(x)
    print(logits.shape)

.. parsed-literal::

   (1, 1000)

More Usage Scenarios
---------------------

In addition to the basic network structure construction, we introduce
the neural network layer (Layer), loss function (Loss) and optimizer
(Optimizer), the parameters (Parameter) required by the neural network
layer and the construction of its initialization method (Initializer),
and other scenarios respectively in detail.

-  `Cell and
   Parameters <https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/modules/layer.html>`__
-  `Parameter
   initialization <https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/modules/initializer.html>`__
-  `Loss
   function <https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/modules/loss.html>`__
-  `Optimizer <https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/modules/optimizer.html>`__

