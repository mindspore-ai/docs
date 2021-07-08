Cell及其子类
===============

概述
-----------

MindSpore的`Cell`类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，需要继承`Cell`类，并重写`__init__`方法和`construct`方法。

损失函数、优化器和模型层等本质上也属于网络结构，也需要继承`Cell`类才能实现功能，同样用户也可以根据业务需求自定义这部分内容。

本节内容首先将会介绍`Cell`类的关键成员函数，然后介绍基于`Cell`实现的MindSpore内置损失函数、优化器和模型层及使用方法，最后通过实例介绍如何利用`Cell`类构建自定义网络。

关键成员函数
--------------

construct方法
^^^^^^^^^^^^^^^^^^^^

`Cell`类重写了`__call__`方法，在`Cell`类的实例被调用时，会执行`construct`方法。网络结构在`construct`方法里面定义。

下面的样例中，我们构建了一个简单的网络实现卷积计算功能。构成网络的算子在`__init__`中定义，在`construct`方法里面使用，用例的网络结构为`Conv2d` -> `BiasAdd`。

在`construct`方法中，`x`为输入数据，`output`是经过网络结构计算后得到的计算结果。

.. code-block::

    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Parameter
    from mindspore.common.initializer import initializer

    class Net(nn.Cell):
        def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
            super(Net, self).__init__()
            self.conv2d = ops.Conv2D(out_channels, kernel_size)
            self.bias_add = ops.BiasAdd()
            self.weight = Parameter(
                initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]),
                name='conv.weight')

        def construct(self, x):
            output = self.conv2d(x, self.weight)
            output = self.bias_add(output, self.bias)
            return output

parameters_dict
^^^^^^^^^^^^^^^

`parameters_dict`方法识别出网络结构中所有的参数，返回一个以key为参数名，value为参数值的`OrderedDict`。

`Cell`类中返回参数的方法还有许多，例如`get_parameters`、`trainable_params`等，具体使用方法可以参见[API文档](https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/nn/mindspore.nn.Cell.html)。

代码样例如下：

.. code-block::

    net = Net()
    result = net.parameters_dict()
    print(result.keys())
    print(result['conv.weight'])

运行结果如下：

.. code-block::

    odict_keys(['conv.weight'])
    Parameter (name=conv.weight)

样例中的`Net`采用上文构造网络的用例，打印了网络中所有参数的名字和`weight`参数的结果。

cells_and_names
^^^^^^^^^^^^^^^

`cells_and_names`方法是一个迭代器，返回网络中每个`Cell`的名字和它的内容本身。

用例简单实现了获取与打印每个`Cell`名字的功能，其中根据网络结构可知，存在1个`Cell`为`nn.Conv2d`。

其中`nn.Conv2d`是`MindSpore`以Cell为基类封装好的一个卷积层，其具体内容将在“模型层”中进行介绍。

代码样例如下：

.. code-block::

    import mindspore.nn as nn

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')

        def construct(self, x):
            out = self.conv(x)
            return out

    net = Net1()
    names = []
    for m in net.cells_and_names():
        print(m)
        names.append(m[0]) if m[0] else None
    print('-------names-------')
    print(names)

运行结果如下：

.. code-block::

    ('', Net1<
    (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=Falseweight_init=normal, bias_init=zeros, format=NCHW>
    >)
    ('conv', Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=Falseweight_init=normal, bias_init=zeros, format=NCHW>)
    -------names-------
    ['conv']

set_grad
^^^^^^^^

`set_grad`接口功能是使用户构建反向网络，在不传入参数调用时，默认设置`requires_grad`为True，需要在计算网络反向的场景中使用。

以`TrainOneStepCell`为例，其接口功能是使网络进行单步训练，需要计算网络反向，因此初始化方法里需要使用`set_grad`。

`TrainOneStepCell`部分代码如下：

.. code-block::

    class TrainOneStepCell(Cell):
        def __init__(self, network, optimizer, sens=1.0):
            super(TrainOneStepCell, self).__init__(auto_prefix=False)
            self.network = network
            self.network.set_grad()
            ......

如果用户使用`TrainOneStepCell`等类似接口无需使用`set_grad`， 内部已封装实现。

若用户需要自定义此类训练功能的接口，需要在其内部调用，或者在外部设置`network.set_grad`。

nn模块与ops模块的关系
----------------------------

MindSpore的nn模块是Python实现的模型组件，是对低阶API的封装，主要包括各种模型层、损失函数、优化器等。

同时nn也提供了部分与`Primitive`算子同名的接口，主要作用是对`Primitive`算子进行进一步封装，为用户提供更友好的API。

重新分析上文介绍`construct`方法的用例，此用例是MindSpore的`nn.Conv2d`源码简化内容，内部会调用`ops.Conv2D`。`nn.Conv2d`卷积API增加输入参数校验功能并判断是否`bias`等，是一个高级封装的模型层。

.. code-block::

    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Parameter
    from mindspore.common.initializer import initializer

    class Net(nn.Cell):
        def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
            super(Net, self).__init__()
            self.conv2d = ops.Conv2D(out_channels, kernel_size)
            self.bias_add = ops.BiasAdd()
            self.weight = Parameter(
                initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]),
                name='conv.weight')

        def construct(self, x):
            output = self.conv2d(x, self.weight)
            output = self.bias_add(output, self.bias)
            return output

.. toctree::
  :maxdepth: 1

  layer
  loss
  optimi
  custom_net