# 网络模型类

`数据处理` `环境准备` `模型导出` `模型训练` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_zh_cn/network_models.md)

<font size=3>**Q：使用MindSpore进行模型训练时，`CTCLoss`的输入参数有四个：`inputs`, `labels_indices`, `labels_values`, `sequence_length`，如何使用`CTCLoss`进行训练？**</font>

A：定义的`model.train`接口里接收的`dataset`可以是多个数据组成，形如(`data1`, `data2`, `data3`, ...)，所以`dataset`是可以包含`inputs`,`labels_indices`,`labels_values`,`sequence_length`的信息的。只需要定义好相应形式的`dataset`，传入`model.train`里就可以。具体的可以了解下相应的[数据处理接口](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/dataset_loading.html)

<br/>

<font size=3>**Q：模型转移时如何把PyTorch的权重加载到MindSpore中？**</font>

A：首先输入PyTorch的`pth`文件，以`ResNet-18`为例，MindSpore的网络结构和PyTorch保持一致，转完之后可直接加载进网络，这边参数只用到`BN`和`Conv2D`，若有其他层`ms`和PyTorch名称不一致，需要同样的修改名称。

<br/>

<font size=3>**Q：模型已经训练好，如何将模型的输出结果保存为文本或者`npy`的格式？**</font>

A：您好，我们网络的输出为`Tensor`，需要使用`asnumpy()`方法将`Tensor`转换为`numpy`，再进行下一步保存。具体可参考：

```python
out = net(x)

np.save("output.npy", out.asnumpy())
```

<br/>

<font size=3>**Q：使用MindSpore做分割训练，必须将数据转为MindRecords吗？**</font>

A：[build_seg_data.py](https://github.com/mindspore-ai/mindspore/blob/r1.1/model_zoo/official/cv/deeplabv3/src/data/build_seg_data.py)是将数据集生成MindRecord的脚本，可以直接使用/适配下你的数据集。或者如果你想尝试自己实现数据集的读取，可以使用`GeneratorDataset`自定义数据集加载。

[GenratorDataset 示例](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/dataset_loading.html#id5)

[GenratorDataset API说明](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

<br/>

<font size=3>**Q：MindSpore可以读取TensorFlow的ckpt文件吗？**</font>

A：MindSpore的`ckpt`和TensorFlow的`ckpt`格式是不通用的，虽然都是使用`protobuf`协议，但是`proto`的定义是不同的。当前MindSpore不支持读取TensorFlow或PyTorch的`ckpt`文件。

<br/>

<font size=3>**Q：如何不将数据处理为MindRecord格式，直接进行训练呢？**</font>

A：可以使用自定义的数据加载方式 `GeneratorDataset`，具体可以参考[数据集加载](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/dataset_loading.html)文档中的自定义数据集加载。

<br/>

<font size=3>**Q：MindSpore现支持直接读取哪些其他框架的模型和哪些格式呢？比如PyTorch下训练得到的pth模型可以加载到MindSpore框架下使用吗？**</font>

A： MindSpore采用protbuf存储训练参数，无法直接读取其他框架的模型。对于模型文件本质保存的就是参数和对应的值，可以用其他框架的API将参数读取出来之后，拿到参数的键值对，然后再加载到MindSpore中使用。比如想用其他框架训练好的ckpt文件，可以先把参数读取出来，再调用MindSpore的`save_checkpoint`接口，就可以保存成MindSpore可以读取的ckpt文件格式了。

<br/>

<font size=3>**Q：用MindSpore训练出的模型如何在Ascend 310上使用？可以转换成适用于HiLens Kit用的吗？**</font>

A：Ascend 310需要运行专用的OM模型,先使用MindSpore导出ONNX或AIR模型，再转化为Ascend 310支持的OM模型。具体可参考[多平台推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/multi_platform_inference_ascend_310.html)。可以，HiLens Kit是以Ascend 310为推理核心，所以前后两个问题本质上是一样的，需要转换为OM模型.

<br/>

<font size=3>**Q：MindSpore如何进行参数（如dropout值）修改？**</font>

A：在构造网络的时候可以通过 `if self.training: x = dropput(x)`，验证的时候，执行前设置`network.set_train(mode_false)`，就可以不适用dropout，训练时设置为True就可以使用dropout。

<br/>

<font size=3>**Q：从哪里可以查看MindSpore训练及推理的样例代码或者教程？**</font>

A：可以访问[MindSpore官网教程训练](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/index.html)和[MindSpore官网教程推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/index.html)。

<br/>

<font size=3>**Q：MindSpore支持哪些模型的训练？**</font>

A：MindSpore针对典型场景均有模型训练支持，支持情况详见[Release note](https://gitee.com/mindspore/mindspore/blob/r1.1/RELEASE.md#)。

<br/>

<font size=3>**Q：MindSpore有哪些现成的推荐类或生成类网络或模型可用？**</font>

A：目前正在开发Wide & Deep、DeepFM、NCF等推荐类模型，NLP领域已经支持Bert_NEZHA，正在开发MASS等模型，用户可根据场景需要改造为生成类网络，可以关注[MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo)。

<br/>

<font size=3>**Q：MindSpore模型训练代码能有多简单？**</font>

A：除去网络定义，MindSpore提供了Model类的接口，大多数场景只需几行代码就可完成模型训练。

<br/>

<font size=3>**Q：如何使用MindSpore拟合$f(x)=a \times sin(x)+b$这类函数？**</font>

A：以下拟合案例是基于MindSpore线性拟合官方案例改编而成。

```python
# The fitting function is：f(x)=2*sin(x)+3.
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

<font size=3>**Q：如何使用MindSpore拟合$f(x)=ax^2+bx+c$这类的二次函数？**</font>

A：以下代码引用自MindSpore的官方教程的[代码仓](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/tutorial_code/linear_regression.py)

在以下几处修改即可很好的拟合$f(x)=ax^2+bx+c$：

1. 数据集生成。
2. 拟合网络。
3. 优化器。

修改的详细信息如下，附带解释。

```python
# Since the selected optimizer does not support CPU, so the training computing platform is changed to GPU, which requires readers to install the corresponding GPU version of MindSpore.
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Assuming that the function to be fitted this time is f(x)=2x^2+3x+4, the data generation function is modified as follows：
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