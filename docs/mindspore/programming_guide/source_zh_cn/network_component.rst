常用网络组件
============

概述
-----

MindSpore封装了一些常用的网络组件，用于网络的训练、推理、求梯度和数据处理等操作。

这些网络组件可以直接被用户使用，同样也会在`model.train`和`model.eval`等更高级的封装接口内部进行使用。

本节内容将会介绍三个网络组件，分别是`GradOperation`、`WithLossCell`和`TrainOneStepCell`，将会从功能、用户使用和内部使用三个方面来进行介绍。

.. toctree::
   :maxdepth: 1

   gradoperation
   withlosscell
   trainonestepcell
