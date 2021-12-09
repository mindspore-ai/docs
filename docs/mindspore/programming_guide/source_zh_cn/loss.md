# 损失函数

`Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [损失函数](#损失函数)
    - [概述](#概述)
    - [内置损失函数](#内置损失函数)
        - [内置损失函数应用实例](#内置损失函数应用实例)
    - [定义损失函数](#定义损失函数)
    - [损失函数与模型训练](#损失函数与模型训练)
        - [定义数据集和网络](#定义数据集和网络)
        - [使用Model进行模型训练](#使用model进行模型训练)
    - [多标签损失函数与模型训练](#多标签损失函数与模型训练)
        - [定义多标签数据集](#定义多标签数据集)
        - [定义多标签损失函数](#定义多标签损失函数)
        - [使用Model进行多标签模型训练](#使用model进行多标签模型训练)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/loss.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

损失函数，又叫目标函数，用于衡量预测值与真实值差异的程度。在深度学习中，模型训练就是通过不停地迭代来缩小损失函数值的过程。因此，在模型训练过程中损失函数的选择非常重要，定义一个好的损失函数，可以有效提高模型的性能。

MindSpore提供了许多通用损失函数供用户选择，但这些通用损失函数并不适用于所有场景，很多情况需要用户自定义所需的损失函数。因此，本教程介绍损失函数的写作方法。

目前MindSpore主要支持的损失函数有`L1Loss`、`MSELoss`、`SmoothL1Loss`、`SoftmaxCrossEntropyWithLogits`、`SampledSoftmaxLoss`、`BCELoss`和`CosineEmbeddingLoss`。

MindSpore的损失函数全部是`Cell`的子类实现，所以也支持用户自定义损失函数，其构造方法在[定义损失函数](#定义损失函数)中进行介绍。

## 内置损失函数

- L1Loss

    计算两个输入数据的绝对值误差，用于回归模型。`reduction`参数默认值为mean，返回loss平均值结果，若`reduction`值为sum，返回loss累加结果，若`reduction`值为none，返回每个loss的结果。

- MSELoss

    计算两个输入数据的平方误差，用于回归模型。`reduction`参数同`L1Loss`。

- SmoothL1Loss

    `SmoothL1Loss`为平滑L1损失函数，用于回归模型，阈值`beta`默认参数为1。

- SoftmaxCrossEntropyWithLogits

    交叉熵损失函数，用于分类模型。当标签数据不是one-hot编码形式时，需要输入参数`sparse`为True。`reduction`参数默认值为none，其参数含义同`L1Loss`。

- CosineEmbeddingLoss

    `CosineEmbeddingLoss`用于衡量两个输入相似程度，用于分类模型。`margin`默认为0.0，`reduction`参数同`L1Loss`。

- BCELoss

    二值交叉熵损失，用于二分类。`weight`是一个batch中每个训练数据的损失的权重，默认值为None，表示权重均为1。`reduction`参数默认值为none，其参数含义同`L1Loss`。
- SampledSoftmaxLoss

   抽样交叉熵损失函数，用于分类模型，一般在类别数很大时使用。`num_sampled`是抽样的类别数，`num_classes`是类别总数，`num_true`是每个用例的类别数，`sampled_values`是默认值为None的抽样候选值。`remove_accidental_hits`是移除“误中抽样”的开关， `seed`是默认值为0的抽样的随机种子，`reduction`参数默认值为none，其参数含义同L1Loss。

### 内置损失函数应用实例

MindSpore的损失函数全部在`mindspore.nn`下，使用方法如下所示：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

loss = nn.L1Loss()
input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
print(loss(input_data, target_data))
```

```text
1.5
```

此用例构造了两个Tensor数据，利用`nn.L1Loss`接口定义了loss，将`input_data`和`target_data`传入loss，执行L1Loss的计算，结果为1.5。若`loss = nn.L1Loss(reduction=’sum’)`，则结果为9.0。若`loss = nn.L1Loss(reduction=’none’)`，结果为`[[1. 0. 2.] [1. 2. 3.]]`。

## 定义损失函数

Cell是MindSpore的基本网络单元，可以用于构建网络，损失函数也需要通过Cell来定义。使用Cell定义损失函数的方法与定义一个普通的网络相同，差别在于，其执行逻辑用于计算前向网络输出与真实值之间的误差。

以MindSpore提供的损失函数L1Loss为例，损失函数的定义方法如下：

```python
import mindspore.nn as nn
import mindspore.ops as ops

class L1Loss(nn.Cell):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.abs = ops.Abs()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.reduce_mean(x)
```

在`__init__`方法中实例化所需的算子，并在`construct`中调用这些算子。这样，一个用于计算L1Loss的损失函数就定义好了。

给定一组预测值和真实值，调用损失函数，就可以得到这组预测值和真实值之间的差异，如下所示：

```python
import numpy as np
from mindspore import Tensor

loss = L1Loss()
input_data = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
target_data = Tensor(np.array([0.1, 0.2, 0.2]).astype(np.float32))

output = loss(input_data, target_data)
print(output)
```

以`Ascend`后端为例，输出结果如下：

```text
0.03333334
```

在定义损失函数时还可以继承损失函数的基类`Loss`。`Loss`提供了`get_loss`方法，用于对损失值求和或求均值，输出一个标量。L1Loss使用`Loss`作为基类的定义如下：

```python
import mindspore.ops as ops
from mindspore.nn import LossBase

class L1Loss(LossBase):
    def __init__(self, reduction="mean"):
        super(L1Loss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x)
```

首先，使用`Loss`作为L1Loss的基类，然后给`__init__`增加一个参数`reduction`，并通过`super`传给基类，最后在`construct`中调用基类提供的`get_loss`方法。`reduction`的合法参数有三个，`mean`、`sum`和`none`，分别表示求均值、求和与输出原值。

## 损失函数与模型训练

接下来使用定义好的L1Loss进行模型训练。

### 定义数据集和网络

这里使用简单的线性拟场景作为样例，数据集和网络结构定义如下：

> 线性拟合详细介绍可参考教程[实现简单线性函数拟合](https://www.mindspore.cn/tutorials/zh-CN/master/linear_regression.html)。

1. 定义数据集

    ```python
    import numpy as np
    from mindspore import dataset as ds

    def get_data(num, w=2.0, b=3.0):
        for _ in range(num):
            x = np.random.uniform(-10.0, 10.0)
            noise = np.random.normal(0, 1)
            y = x * w + b + noise
            yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

    def create_dataset(num_data, batch_size=16):
        dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
        dataset = dataset.batch(batch_size)
        return dataset
    ```

2. 定义网络

    ```python
    from mindspore.common.initializer import Normal
    import mindspore.nn as nn

    class LinearNet(nn.Cell):
        def __init__(self):
            super(LinearNet, self).__init__()
            self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

        def construct(self, x):
            return self.fc(x)
    ```

### 使用Model进行模型训练

`Model`是MindSpore提供的用于模型训练、评估和推理的高阶API。创建数据集并定义一个`Model`就可以使用`train`接口进行模型训练。接下来我们使用`Model`进行模型训练，并采用之前定义好的`L1Loss`作为此次训练的损失函数。

1. 定义前向网络、损失函数和优化器

    使用之前定义的`LinearNet`和`L1Loss`作为前向网络和损失函数，并选择MindSpore提供的`Momemtum`作为优化器。

    ```python
    # define network
    net = LinearNet()
    # define loss function
    loss = L1Loss()
    # define optimizer
    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    ```

2. 定义`Model`

    定义`Model`时需要指定前向网络、损失函数和优化器，`Model`内部会将它们关联起来，组成一张训练网。

    ```python
    from mindspore import Model

    # define Model
    model = Model(net, loss, opt)
    ```

3. 创建数据集，并调用`train`接口进行模型训练

    调用`train`接口时必须指定迭代次数`epoch`和训练数据集`train_dataset`，我们将`epoch`设置为1，将`create_dataset`创建的数据集作为训练集。`callbacks`是`train`接口的可选参数，在`callbacks`中使用`LossMonitor`可以监控训练过程中损失函数值的变化。`dataset_sink_mode`也是一个可选参数，这里设置为`False`，表示使用非下沉模式进行训练。

    ```python
    from mindspore.train.callback import LossMonitor

    # create dataset
    ds_train = create_dataset(num_data=160)
    # training
    model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
    ```

完整代码如下：

> 下述例子中，参数初始化使用了随机值，在具体执行中输出的结果可能与本地执行输出的结果不同；如果需要稳定输出固定的值，可以设置固定的随机种子，设置方法请参考[mindspore.set_seed()](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)。

```python
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model
from mindspore import dataset as ds
from mindspore.nn import LossBase
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

class L1Loss(LossBase):
    def __init__(self, reduction="mean"):
        super(L1Loss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x)

def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16):
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset

# define network
net = LinearNet()
# define loss functhon
loss = L1Loss()
# define optimizer
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# define Model
model = Model(net, loss, opt)
# create dataset
ds_train = create_dataset(num_data=160)
# training
model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

执行结果如下：

```text
epoch: 1 step: 1, loss is 8.328788
epoch: 1 step: 2, loss is 8.594973
epoch: 1 step: 3, loss is 13.299595
epoch: 1 step: 4, loss is 9.04059
epoch: 1 step: 5, loss is 8.991402
epoch: 1 step: 6, loss is 6.5928526
epoch: 1 step: 7, loss is 8.239887
epoch: 1 step: 8, loss is 7.3984795
epoch: 1 step: 9, loss is 7.33724
epoch: 1 step: 10, loss is 4.3588376
```

## 多标签损失函数与模型训练

上一章定义了一个简单的损失函数`L1Loss`，其他损失函数可以仿照`L1Loss`进行编写。但许多深度学习应用的数据集较复杂，例如目标检测网络Faster R-CNN的数据中就包含多个标签，而不是简单的data和label，这时候损失函数的定义和使用略有不同。

Faster R-CNN网络结构较复杂，不便在此处详细展开。本章对上一章中描述的线性拟合场景进行扩展，手动构建一个多标签数据集，介绍在这种场景下如何定义损失函数，并通过`Model`进行训练。

### 定义多标签数据集

首先定义数据集。对之前定义的数据集稍作修改：

1. `get_multilabel_data`中产生两个标签`y1`和`y2`
2. `GeneratorDataset`的`column_names`参数设置为['data', 'label1', 'label2']

这样通过`create_multilabel_dataset`产生的数据集就有一个数据`data`，两个标签`label1`和`label2`。

```python
import numpy as np
from mindspore import dataset as ds

def get_multilabel_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise1 = np.random.normal(0, 1)
        noise2 = np.random.normal(-1, 1)
        y1 = x * w + b + noise1
        y2 = x * w + b + noise2
        yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)

def create_multilabel_dataset(num_data, batch_size=16):
    dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
    dataset = dataset.batch(batch_size)
    return dataset
```

### 定义多标签损失函数

针对上一步创建的数据集，定义损失函数`L1LossForMultiLabel`。此时，损失函数`construct`的输入有三个，预测值`base`，真实值`target1`和`target2`，我们在`construct`中分别计算预测值与真实值`target1`、`target2`之间的误差，将这两个误差的均值作为最终的损失函数值，具体如下：

```python
import mindspore.ops as ops
from mindspore.nn import LossBase

class L1LossForMultiLabel(LossBase):
    def __init__(self, reduction="mean"):
        super(L1LossForMultiLabel, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target1, target2):
        x1 = self.abs(base - target1)
        x2 = self.abs(base - target2)
        return self.get_loss(x1)/2 + self.get_loss(x2)/2
```

### 使用Model进行多标签模型训练

刚才提到过，Model内部会关联用户指定的前向网络、损失函数和优化器。其中，前向网络和损失函数是通过`nn.WithLossCell`关联起来的，`nn.WithLossCell`会将前向网络和损失函数连接起来，如下：

```python
import mindspore.nn as nn

class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        output = self._backbone(data)
        return self._loss_fn(output, label)
```

注意到`Model`默认使用的`nn.WithLossCell`只有两个输入，`data`和`label`，对于多个标签的场景显然不适用。此时，如果想要使用`Model`进行模型训练就需要用户将前向网络与损失函数连接起来，具体如下：

1. 定义适用于当前场景的`CustomWithLossCell`

    仿照`nn.WithLossCell`进行定义，将`construct`的输入修改为三个，将数据部分传给`backend`，将预测值和两个标签传给`loss_fn`。

    ```python
    import mindspore.nn as nn

    class CustomWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(CustomWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label1, label2):
            output = self._backbone(data)
            return self._loss_fn(output, label1, label2)
    ```

2. 使用`CustomWithLossCell`将前向网络和损失函数连接起来

    前向网络使用上一章定义的`LinearNet`，损失函数使用`L1LossForMultiLabel`，用`CustomWithLossCell`将它们连接起来，如下：

    ```python
    net = LinearNet()
    loss = L1LossForMultiLabel()
    loss_net = CustomWithLossCell(net, loss)
    ```

    这样`loss_net`中就包含了前向网络和损失函数的运算逻辑。

3. 定义Model并进行模型训练

    `Model`的`network`指定为`loss_net`，`loss_fn`不指定，优化器仍使用`Momentum`。此时用户未指定`loss_fn`，`Model`则认为`network`内部已经实现了损失函数的逻辑，便不会用`nn.WithLossCell`对前向函数和损失函数进行封装。

    使用`create_multilabel_dataset`创建多标签数据集并进行训练：

    ```python
    from mindspore.train.callback import LossMonitor
    from mindspore import Model

    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    model = Model(network=loss_net, optimizer=opt)
    ds_train = create_multilabel_dataset(num_data=160)
    model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
    ```

完整代码如下：

> 下述例子中，参数初始化使用了随机值，在具体执行中输出的结果可能与本地执行输出的结果不同；如果需要稳定输出固定的值，可以设置固定的随机种子，设置方法请参考[mindspore.set_seed()](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)。

```python
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model
from mindspore import dataset as ds
from mindspore.nn import LossBase
from mindspore.common.initializer import Normal
from mindspore.train.callback import LossMonitor

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

class L1LossForMultiLabel(LossBase):
    def __init__(self, reduction="mean"):
        super(L1LossForMultiLabel, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target1, target2):
        x1 = self.abs(base - target1)
        x2 = self.abs(base - target2)
        return self.get_loss(x1)/2 + self.get_loss(x2)/2

class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2):
        output = self._backbone(data)
        return self._loss_fn(output, label1, label2)

def get_multilabel_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise1 = np.random.normal(0, 1)
        noise2 = np.random.normal(-1, 1)
        y1 = x * w + b + noise1
        y2 = x * w + b + noise2
        yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)

def create_multilabel_dataset(num_data, batch_size=16):
    dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
    dataset = dataset.batch(batch_size)
    return dataset

net = LinearNet()
loss = L1LossForMultiLabel()
# build loss network
loss_net = CustomWithLossCell(net, loss)

opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
model = Model(network=loss_net, optimizer=opt)
ds_train = create_multilabel_dataset(num_data=160)
model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

执行结果如下：

```text
epoch: 1 step: 1, loss is 11.039986
epoch: 1 step: 2, loss is 7.7847576
epoch: 1 step: 3, loss is 9.236277
epoch: 1 step: 4, loss is 8.3316345
epoch: 1 step: 5, loss is 6.957058
epoch: 1 step: 6, loss is 9.231144
epoch: 1 step: 7, loss is 9.1072
epoch: 1 step: 8, loss is 6.7703295
epoch: 1 step: 9, loss is 6.363703
epoch: 1 step: 10, loss is 5.014839
```

本章节简单讲解了多标签数据集场景下，如何定义损失函数并使用Model进行模型训练。在很多其他场景中，也可以采用此类方法进行模型训练。
