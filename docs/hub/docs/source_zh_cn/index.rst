MindSpore Hub 文档
=========================

MindSpore Hub是MindSpore生态的预训练模型应用工具。

MindSpore Hub包含以下功能：

- 即插即用的模型加载
- 简单易用的迁移学习

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

代码仓地址： <https://gitee.com/mindspore/hub>

使用MindSpore Hub的典型场景
----------------------------

1. `推理验证 <https://www.mindspore.cn/hub/docs/zh-CN/master/loading_model_from_hub.html#用于推理验证>`_

   mindspore_hub.load用于加载预训练模型，可以实现一行代码完成模型的加载。

2. `迁移学习 <https://www.mindspore.cn/hub/docs/zh-CN/master/loading_model_from_hub.html#用于迁移学习>`_

   通过mindspore_hub.load完成模型加载后，可以增加一个额外的参数项只加载神经网络的特征提取部分，这样就能很容易地在之后增加一些新的层进行迁移学习。

3. `发布模型 <https://www.mindspore.cn/hub/docs/zh-CN/master/publish_model.html>`_

   可以将自己训练好的模型按照指定的步骤发布到MindSpore Hub中，以供其他用户进行下载和使用。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   hub_installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   loading_model_from_hub
   publish_model

.. toctree::
   :maxdepth: 1
   :caption: API参考

   hub

.. toctree::
   :maxdepth: 1
   :caption: 模型

   MindSpore Hub↗ <https://www.mindspore.cn/hub>
