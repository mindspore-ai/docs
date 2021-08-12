﻿# 执行问题

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/faq/source_zh_cn/implement_problem.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

<font size=3>**Q: 使用MindSpore在模型保存后生成的`.meta`文件作用是什么，可以用`.meta`文件导入图结构吗？**</font>

A: 这里的`.meta`文件是编译好的图结构，但是目前并不支持直接导入这种结构。如果不知道图结构的情况下想要导入网络，还是需要用MindIR格式的文件。

<br/>

<font size=3>**Q: 请问`yolov4-tiny-3l.weights`模型文件可以直接转换成MindSpore模型吗？**</font>

A: 不能的，需要把其他框架训练好的参数转换成MindSpore的格式，才能转成MindSpore的模型。

<br/>

<font size=3>**Q: 使用MindSpore进行`model.train`的时候进行了如下设置，为什么会报错呢？**</font>

```python
model.train(1, dataset, callbacks=LossMonitor(1), dataset_sink_mode=True)
model.train(1, dataset, callbacks=LossMonitor(1), dataset_sink_mode=False)
```

A: 因为在已经设置为下沉模式的情况下，就不能再设置为非下沉了，是运行机制上的限制。

<br/>

<font size=3>**Q: 使用MindSpore训练模型在`eval`阶段，需要注意什么？能够直接加载网络和参数吗？需要在Model中使用优化器吗？**</font>

A: 在`eval`阶段主要看需要什么，比如图像分类任务`eval`网络的输出是各个类的概率值，与对应标签计算`acc`。
大多数情况是可以直接复用训练的网络和参数的，需要注意的是需要设置推理模式。

```python
net.set_train(False)
```

在eval阶段不需要优化器，但是需要使用MindSpore的`model.eval`接口的话需要配置一下`loss function`，如：

```python
# 定义模型
model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
# 评估模型
res = model.eval(dataset)
```

<br/>

<font size=3>**Q: 如何使用SGD里的`param_group`来实现学习率的衰减？**</font>

A: 如果需要按照`epoch`来变化，可以使用[Dynamic LR](https://mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.nn.html#dynamic-lr),把其中的`step_per_epoch`设置成`step_size`，如果需要按照`step`来变化，可以把其中的`step_per_epoch`设置成1，也可以用[LearningRateSchedule](https://mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.nn.html#dynamic-learning-rate)。

<br/>

<font size=3>**Q: MindSpore如何进行参数（如dropout值）修改？**</font>

A: 在构造网络的时候可以通过 `if self.training: x = dropput(x)`，推理时，执行前设置`network.set_train(mode_false)`，就可以不使用dropout，训练时设置为True就可以使用dropout。

<br/>

<font size=3>**Q: 如何查看模型参数量？**</font>

A: 可以直接加载CheckPoint统计，可能额外统计了动量和optimizer中的变量，需要过滤下相关变量。
您可以参考如下接口统计网络参数量:

```python
def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params
```

具体[脚本链接](https://gitee.com/mindspore/mindspore/blob/r1.3/model_zoo/research/cv/tinynet/src/utils.py)。

<br/>

<font size=3>**Q: 如何在训练过程中监控`loss`在最低的时候并保存训练参数？**</font>

A: 可以自定义一个`Callback`。参考`ModelCheckpoint`的写法，此外再增加判断`loss`的逻辑:

```python
class EarlyStop(Callback):
    def __init__(self):
        self.loss = None
    def step_end(self, run_context):
        loss =  ****(get current loss)
        if (self.loss == None or loss < self.loss):
            self.loss = loss
            # do save ckpt
```

<br/>

<font size=3>**Q: 使用`nn.Conv2d`时，怎样获取期望大小的`feature map`？**</font>

A: `Conv2d shape`推导方法可以[参考这里](https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d)，`Conv2d`的`pad_mode`改成`same`，或者可以根据`Conv2d shape`推导公式自行计算`pad`，想要使得`shape`不变，一般pad为`(kernel_size-1)//2`。

<br/>

<font size=3>**Q: 使用MindSpore可以自定义一个可以返回多个值的loss函数？**</font>

A: 自定义`loss function`后还需自定义`TrainOneStepCell`，实现梯度计算时`sens`的个数和`network`的输出个数相同。具体可参考:

```python
net = Net()
loss_fn = MyLoss()
loss_with_net = MyWithLossCell(net, loss_fn)
train_net = MyTrainOneStepCell(loss_with_net, optim)
model = Model(net=train_net, loss_fn=None, optimizer=None)
```

<br/>

<font size=3>**Q: MindSpore如何实现早停功能？**</font>

A: 可以自定义`callback`方法实现早停功能。
例子: 当loss降到一定数值后，停止训练。

```python
class EarlyStop(Callback):
    def __init__(self, control_loss=1):
        super(EarlyStep, self).__init__()
        self._control_loss = control_loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if loss.asnumpy() < self._control_loss:
            # Stop training
            run_context._stop_requested = True

stop_cb = EarlyStop(control_loss=1)
model.train(epoch_size, ds_train, callbacks=[stop_cb])
```

<br/>

<font size=3>**Q: 模型已经训练好，如何将模型的输出结果保存为文本或者`npy`的格式？**</font>

A: 您好，我们网络的输出为`Tensor`，需要使用`asnumpy()`方法将`Tensor`转换为`numpy`，再进行下一步保存。具体可参考:

```python
out = net(x)
np.save("output.npy", out.asnumpy())
```

<br/>

<font size=3>**Q: 运行时报错“Create python object \`<class 'mindspore.common.tensor.Tensor'>\` failed, only support create Cell or Primitive object.”怎么办？**</font>

A: 当前在图模式下，`construct`函数(或`@ms_function`装饰器修饰的函数)仅支持构造`Cell`和`Primitive object`，不支持构造`Tensor`，即不支持语法`x = Tensor(args...)`。

如果是常量`Tensor`，请在`__init__`函数中定义。如果不是常量`Tensor`，可以通过`@constexpr`装饰器修饰函数，在函数里生成`Tensor`。

关于`@constexpr`的用法可参考: <https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/ops/mindspore.ops.constexpr.html>。

对于网络中需要用到的常量`Tensor`，可以作为网络的属性，在`init`的时候定义，即`self.x = Tensor(args...)`，然后在`construct`函数(或`@ms_function`装饰器修饰的函数)里使用。

如下示例，通过`@constexpr`生成一个`shape = (3, 4), dtype = int64`的`Tensor`。

```python
@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4).astype(np.int64)))
```

<br/>

<font size=3>**Q: 运行时报错“'self.xx' should be defined in the class '__init__' function.”怎么办？**</font>

A: 如果在`construct`函数里，想对类成员`self.xx`赋值，那么`self.xx`必须已经在`__init__`函数中被定义为[`Parameter`](<https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.html#mindspore.Parameter>)类型，其他类型则不支持。局部变量`xx`不受这个限制。

<br/>

<font size=3>**Q: 运行时报错“This comparator 'AnyValue' is not supported. For statement 'is', only support compare with 'None', 'False' or 'True'”怎么办？**</font>

A: 对于语法`is` 或 `is not`而言，当前`MindSpore`仅支持与`True`、`False`和`None`的比较。暂不支持其他类型，如字符串等。

<br/>

<font size=3>**Q: 运行时报错“MindSpore does not support comparison with operators more than one now, ops size =2”怎么办？**</font>

A: 对于比较语句，`MindSpore`最多支持一个操作数。例如不支持语句`1 < x < 3`，请使用`1 < x and x < 3`的方式代替。

<br/>

<font size=3>**Q: 运行时报错“TypeError: The function construct need 1 positional argument and 0 default argument, but provided 2”怎么办？**</font>

A: 网络的实例被调用时，会执行`construct`方法，然后会检查`construct`方法需要的参数个数和实际传入的参数个数，如果不一致则会抛出以上异常。
请检查脚本中调用网络实例时传入的参数个数，和定义的网络中`construct`函数需要的参数个数是否一致。

<br/>

<font size=3>**Q: 运行时报错“Type Join Failed”或“Shape Join Failed”怎么办？**</font>

A: 在前端编译的推理阶段，会对节点的抽象类型(包含`type`、`shape`等)进行推导，常见抽象类型包括`AbstractScalar`、`AbstractTensor`、`AbstractFunction`、`AbstractTuple`、`AbstractList`等。在一些场景比如多分支场景，会对不同分支返回值的抽象类型进行`join`合并，推导出返回结果的抽象类型。如果抽象类型不匹配，或者`type`/`shape`不一致，则会抛出以上异常。

当出现类似“Type Join Failed: dtype1 = Float32, dtype2 = Float16”的报错时，说明数据类型不一致，导致抽象类型合并失败。根据提供的数据类型和代码行信息，可以快速定位出错范围。此外，报错信息中提供了具体的抽象类型信息、节点信息，可以通过`analyze_fail.dat`文件查看MindIR信息，定位解决问题。关于MindIR的具体介绍，可以参考[MindSpore IR（MindIR）](https://www.mindspore.cn/docs/note/zh-CN/r1.3/design/mindir.html)。代码样例如下:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, context

context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.cast = ops.Cast()

    def construct(self, x, a, b):
        if a > b:
            return self.relu(x)
        else:
            return self.cast(self.relu(x), ms.float16)

input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = Tensor(2, ms.float32)
input_b = Tensor(6, ms.float32)
net = Net()
out_me = net(input_x, input_a, input_b)
```

执行结果如下:

```text
TypeError: The return values of different branches do not match. Type Join Failed: dtype1 = Float32, dtype2 = Float16. The abstract type of the return value of the current branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float16, Value: AnyValue, Shape: NoShape), value_ptr: 0x32ed00e0, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x32ed00e0, value: AnyValue). Please check the node construct.4:[CNode]5{[0]: [CNode]6}, true branch: ✓construct.2, false branch: ✗construct.3. trace:
In file test_type_join_failed.py(14)/        if a > b:/

The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test_type_join_failed.py(14)
        if a > b:
```

当出现类似“Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ()”的报错时，说明`shape`不一致，导致抽象类型合并失败。代码样例如下:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, context

context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.reducesum = ops.ReduceSum()

    def construct(self, x, a, b):
        if a > b:
            return self.relu(x)
        else:
            return self.reducesum(x)

input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = Tensor(2, ms.float32)
input_b = Tensor(6, ms.float32)
net = Net()
out = net(input_x, input_a, input_b)
```

执行结果如下:

```text
ValueError: The return values of different branches do not match. Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = (). The abstract type of the return value of the current branch is AbstractTensor(shape: (), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x239b5120, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x239b5120, value: AnyValue). Please check the node construct.4:[CNode]5{[0]: [CNode]6}, true branch: ✓construct.2, false branch: ✗construct.3. trace:
In file test_shape_join_failed.py(14)/        if a > b:/

The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test_shape_join_failed.py(14)
        if a > b:
```

当出现如“Type Join Failed: abstract type AbstractTensor can not join with AbstractTuple”的报错时，说明这两种抽象类型无法匹配，需要根据提供的代码行等报错信息，重新检视代码并修改。

<br/>

<font size=3>**Q: 缓存服务器异常关闭如何处理？**</font>

A: 缓存服务器使用过程中，会进行IPC共享内存和socket文件等系统资源的分配。若允许溢出，在磁盘空间还会存在溢出的数据文件。一般情况下，如果通过`cache_admin --stop`命令正常关闭服务器，这些资源将会被自动清理。

但如果缓存服务器被异常关闭，例如缓存服务进程被杀等，用户需要首先尝试重新启动服务器，若启动失败，则应该依照以下步骤手动清理系统资源:

- 删除IPC资源。

    1. 检查是否有IPC共享内存残留。

        一般情况下，系统会为缓存服务分配4GB的共享内存。通过以下命令可以查看系统中的共享内存块使用情况。

        ```text
        $ ipcs -m
        ------ Shared Memory Segments --------
        key        shmid      owner      perms      bytes      nattch     status
        0x61020024 15532037   root       666        4294967296 1
        ```

        其中，`shmid`为共享内存块id，`bytes`为共享内存块的大小，`nattch`为链接到该共享内存块的进程数量。`nattch`不为0表示仍有进程使用该共享内存块。在删除共享内存前，需要停止使用该内存块的所有进程。

    2. 删除IPC共享内存。

        找到对应的共享内存id，并通过以下命令删除。

        ```text
        ipcrm -m {shmid}
        ```

- 删除socket文件。

    一般情况下，socket文件位于`/tmp/mindspore/cache`。进入文件夹，执行以下命令删除socket文件。

    ```text
    rm cache_server_p{port_number}
    ```

    其中`port_number`为用户创建缓存服务器时指定的端口号，默认为50052。

- 删除溢出到磁盘空间的数据文件。

    进入启用缓存服务器时指定的溢出数据路径。通常，默认溢出路径为`/tmp/mindspore/cache`。找到路径下对应的数据文件夹并逐一删除。

<br/>

<font size=3>**Q: 通过Hub可以使用GPU加载`vgg16`模型以及是否可以做迁移模型吗？**</font>

A: 请手动修改如下两处参数即可:

```python
# 增加**kwargs参数: 如下
def vgg16(num_classes=1000, args=None, phase="train", **kwargs):
```

```python
# 增加**kwargs参数: 如下
net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase, **kwargs)
```

<br/>

<font size=3>**Q: 如何得到VGG模型中间层特征？**</font>

A: 你好，获取网络中间层的特征，其实跟具体框架没有太大关系了。`torchvison`里定义的`vgg`模型，可以通过`features`字段获取"中间层特征"，`torchvison`的`vgg`源码如下:

```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
```

在MindSpore的ModelZoo里定义的`vgg16`，可以通过`layers`字段获取，如下:

```python
network = vgg16()
print(network.layers)
```

<br/>

<font size=3>**Q: 使用MindSpore进行模型训练时，`CTCLoss`的输入参数有四个: `inputs`, `labels_indices`, `labels_values`, `sequence_length`，如何使用`CTCLoss`进行训练？**</font>

A: 定义的`model.train`接口里接收的`dataset`可以是多个数据组成，形如(`data1`, `data2`, `data3`, ...)，所以`dataset`是可以包含`inputs`,`labels_indices`,`labels_values`,`sequence_length`的信息的。只需要定义好相应形式的`dataset`，传入`model.train`里就可以。具体的可以了解下相应的[数据处理接口](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/dataset_loading.html)

<br/>

<font size=3>**Q: 模型转移时如何把PyTorch的权重加载到MindSpore中？**</font>

A: 首先输入PyTorch的`pth`文件，以`ResNet-18`为例，MindSpore的网络结构和PyTorch保持一致，转完之后可直接加载进网络，这边参数只用到`BN`和`Conv2D`，若有其他层`ms`和PyTorch名称不一致，需要同样的修改名称。

<br/>

<font size=3>**Q: MindSpore有哪些现成的推荐类或生成类网络或模型可用？**</font>

A: 目前正在开发Wide & Deep、DeepFM、NCF等推荐类模型，NLP领域已经支持Bert_NEZHA，正在开发MASS等模型，用户可根据场景需要改造为生成类网络，可以关注[MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo)。

<br/>

<font size=3>**Q: 如何使用MindSpore拟合$f(x)=a \times sin(x)+b$这类函数？**</font>

A: 以下拟合案例是基于MindSpore线性拟合官方案例改编而成。

```python
# The fitting function is: f(x)=2*sin(x)+3.
import numpy as np
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import nn, Model, context
from mindspore.train.callback import LossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

 def get_data(num, w=2.0, b=3.0):
    # f(x)=w * sin(x) + b
    # f(x)=2 * sin(x) +3
    for i in range(num):
        x = np.random.uniform(-np.pi, np.pi)
        noise = np.random.normal(0, 1)
        y = w * np.sin(x) + b + noise
        yield np.array([np.sin(x)]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    num_data = 1600
    batch_size = 16
    repeat_size = 1
    lr = 0.005
    momentum = 0.9

    net = LinearNet()
    net_loss = nn.loss.MSELoss()
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    model = Model(net, net_loss, opt)

    ds_train = create_dataset(num_data, batch_size=batch_size, repeat_size=repeat_size)

    model.train(1, ds_train, callbacks=LossMonitor(), dataset_sink_mode=False)

    print(net.trainable_params()[0], "\n%s" % net.trainable_params()[1])
```

<br/>

<font size=3>**Q: 如何使用MindSpore拟合$f(x)=ax^2+bx+c$这类的二次函数？**</font>

A: 以下代码引用自MindSpore的官方教程的[代码仓](https://gitee.com/mindspore/docs/blob/r1.3/docs/sample_code/linear_regression.py)

在以下几处修改即可很好的拟合$f(x)=ax^2+bx+c$:

1. 数据集生成。
2. 拟合网络。
3. 优化器。

修改的详细信息如下，附带解释。

```python
# Since the selected optimizer does not support CPU, so the training computing platform is changed to GPU, which requires readers to install the corresponding GPU version of MindSpore.
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Assuming that the function to be fitted this time is f(x)=2x^2+3x+4, the data generation function is modified as follows:
def get_data(num, a=2.0, b=3.0 ,c = 4):
    for i in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        # The y value is generated by the fitting target function ax^2+bx+c.
        y = x * x * a + x * b + c + noise
        # When a*x^2+b*x+c is fitted, a and b are weight parameters and c is offset parameter bias. The training data corresponding to the two weights are x^2 and x respectively, so the data set generation mode  is changed as follows:
        yield np.array([x*x, x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        # Because the full join function inputs two training parameters, the input value is changed to 2, the first Nomral(0.02) will automatically assign random weights to the input two parameters, and the second Normal is the random bias.
        self.fc = nn.Dense(2, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    num_data = 1600
    batch_size = 16
    repeat_size = 1
    lr = 0.005
    momentum = 0.9

    net = LinearNet()
    net_loss = nn.loss.MSELoss()
    # RMSProp optimalizer with better effect is selected for quadratic function fitting, Currently, Ascend and GPU computing platforms are supported.
    opt = nn.RMSProp(net.trainable_params(), learning_rate=0.1)
    model = Model(net, net_loss, opt)

    ds_train = create_dataset(num_data, batch_size=batch_size, repeat_size=repeat_size)
    model.train(1, ds_train, callbacks=LossMonitor(), dataset_sink_mode=False)

    print(net.trainable_params()[0], "\n%s" % net.trainable_params()[1])
```

<br/>

<font size=3>**Q: `mindspore/tests`下怎样执行单个`ut`用例？**</font>

A: `ut`用例通常需要基于debug版本的MindSpore包，官网并没有提供。可以基于源码使用`sh build.sh`编译，然后通过`pytest`指令执行，debug模式编包不依赖后端。编译选项`sh build.sh -t on`，用例执行可以参考`tests/runtest.sh`脚本。

<br/>

<font size=3>**Q: 在Ascend平台上，执行用例有时候会报错`run task error`，如何获取更详细的日志帮助问题定位？**</font>

A: 使用msnpureport工具设置device侧日志级别，工具位置在: `/usr/local/Ascend/driver/tools/msnpureport`。

- 全局级别:

```bash
/usr/local/Ascend/driver/tools/msnpureport -g info
```

- 模块级别:

```bash
/usr/local/Ascend/driver/tools/msnpureport -m SLOG:error
````

- Event级别:

```bash
/usr/local/Ascend/driver/tools/msnpureport -e disable/enable
```

- 多device id级别:

```bash
/usr/local/Ascend/driver/tools/msnpureport -d 1 -g warning
```

假设deviceID的取值范围是[0-7]，`device0`-`device3`和`device4`-`device7`分别在一个os上。其中`device0`-`device3`共用一个日志配置文件；`device4`-`device7`共用一个配置文件。如果修改了`device0`-`device3`中的任意一个日志级别，其他`device`的日志级别也会被修改。如果修改了`device4`-`device7`中的任意一个日志级别，其他device的日志级别也会被修改。

`Driver`包安装以后（假设安装路径为/usr/local/HiAI，在Windows环境下，`msnpureport.exe`执行文件在C:\ProgramFiles\Huawei\Ascend\Driver\tools\目录下），假设用户在/home/shihangbo/目录下直接执行命令行，则Device侧日志被导出到当前目录下，并以时间戳命名文件夹进行存放。

<br/>

<font size=3>**Q: 使用Ascend平台执行训练过程，出现报错: `Out of Memory!!! total[3212254720] (dynamic[0] memory poll[524288000]) malloc[32611480064] failed!` 如何解决？**</font>

A: 此问题属于内存占用过多导致的内存不够问题，可能原因有两种:

- `batch_size`的值设置过大。解决办法: 将`batch_size`的值设置减小。
- 引入了异常大的`Parameter`，例如单个数据shape为[640,1024,80,81]，数据类型为float32，单个数据大小超过15G，这样差不多大小的两个数据相加时，占用内存超过3*15G，容易造成`Out of Memory`。解决办法: 检查参数的`shape`，如果异常过大，减少shape。
- 如果以上操作还是未能解决，可以上[官方论坛](https://bbs.huaweicloud.com/forum/forum-1076-1.html)发帖提出问题，将会有专门的技术人员帮助解决。

<br/>

<font size=3>**Q: 如何在训练神经网络过程中对计算损失的超参数进行改变？**</font>

A: 您好，很抱歉暂时还未有这样的功能。目前只能通过训练-->重新定义优化器-->训练，这样的过程寻找较优的超参数。

<br/>

<font size=3>**Q: 运行应用时报错`error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A: 安装MindSpore所依赖的Ascend 310 AI处理器配套软件包时，`CANN`包不能安装`nnrt`版本，而是需要安装功能完整的`toolkit`版本。

<br/>

<font size=3>**Q: MindSpore代码里面的model_zoo/official/cv/resnet/train.py中context.set_ps_context(enable_ps=True)为什么一定要在init之前设置**</font>

A: MindSpore Ascend模式下，如果先调用init，那么会为所有的进程都分配卡，但是parameter server训练模式下server是不需要分配卡的，那么worker和server就会去使用同一块卡，导致会报错: Hccl dependent tsd is not open。
