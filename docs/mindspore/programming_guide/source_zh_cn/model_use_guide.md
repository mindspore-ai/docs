# Model基本使用

`Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [Model基本使用](#model基本使用)
    - [概述](#概述)
    - [Model基本介绍](#model基本介绍)
    - [模型训练、评估和推理](#模型训练评估和推理)
    - [自定义场景的Model应用](#自定义场景的model应用)
        - [手动连接前向网络与损失函数](#手动连接前向网络与损失函数)
        - [自定义训练网络](#自定义训练网络)
        - [自定义网络的权重共享](#自定义网络的权重共享)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/model_use_guide.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

[编程指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/index.html)的网络构建部分讲述了如何定义前向网络、损失函数和优化器，并介绍了如何将这些结构封装成训练、评估网络并执行。在此基础上，本文档讲述如何使用高阶API`Model`进行模型训练和评估。

通常情况下，定义训练和评估网络并直接运行，已经可以满足基本需求，但仍然建议通过`Model`来进行模型训练和评估。一方面，`Model`可以在一定程度上简化代码。例如：无需手动遍历数据集；在不需要自定义`TrainOneStepCell`的场景下，可以借助`Model`自动构建训练网络；可以使用`Model`的`eval`接口进行模型评估，直接输出评估结果，无需手动调用评价指标的`clear`、`update`、`eval`函数等。另一方面，`Model`提供了很多高阶功能，如数据下沉、混合精度等，在不借助`Model`的情况下，使用这些功能需要花费较多的时间仿照`Model`进行自定义。

本文档首先对Model进行基本介绍，然后重点讲解如何使用`Model`进行模型训练、评估和推理。

> 下述例子中，参数初始化使用了随机值，在具体执行中输出的结果可能与本地执行输出的结果不同；如果需要稳定输出固定的值，可以设置固定的随机种子，设置方法请参考[mindspore.set_seed()](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore/mindspore.set_seed.html)。

## Model基本介绍

`Model`是MindSpore提供的模型训练高阶API，可以进行模型训练、评估和推理。

`Model`中包含入参：

- network (Cell)：一般情况下为前向网络，输入数据和标签，输出预测值。

- loss_fn (Cell)：所使用的损失函数。

- optimizer (Cell)：所使用的优化器。

- metrics (set)：进行模型评估时使用的评价指标，在不需要模型评估时使用默认值`None`。

- eval_network (Cell)：模型评估所使用的网络，在部分简单场景下不需要指定。

- eval_indexes (List)：用于指示评估网络输出的含义，配合`eval_network`使用，该参数的功能可通过`nn.Metric`的`set_indexes`代替，建议使用`set_indexes`。

- amp_level (str)：用于指定混合精度级别。

- kwargs：可配置溢出检测和混合精度策略。

`Model`提供了以下接口用于模型训练、评估和推理：

- train：用于在训练集上进行模型训练。

- eval：用于在验证集上进行模型评估。

- predict：用于对输入的一组数据进行推理，输出预测结果。

## 模型训练、评估和推理

对于简单场景的神经网络，可以在定义`Model`时指定前向网络`network`、损失函数`loss_fn`、优化器`optimizer`和评估指标`metrics`。此时，Model会使用`network`作为推理网络，并使用`nn.WithLossCell`和`nn.TrainOneStepCell`构建训练网络，使用`nn.WithEvalCell`构建评估网络。

以[构建训练与评估网络](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/train_and_eval.html)中使用的线性回归为例：

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

net = LinearNet()
# 设定损失函数
crit = nn.MSELoss()
# 设定优化器
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# 设定评价指标
metrics = {"mae"}
```

[构建训练与评估网络](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/train_and_eval.html)中讲述了通过`nn.WithLossCell`、`nn.TrainOneStepCell`和`nn.WithEvalCell`构建训练和评估网络并直接运行方式。使用`Model`时则不需要手动构建训练和评估网络，用以下方式定义`Model`并调用`train`和`eval`接口能够达到相同的效果。

创建训练集和验证集：

```python
import numpy as np
import mindspore.dataset as ds

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

# 创建数据集
train_dataset = create_dataset(num_data=160)
eval_dataset = create_dataset(num_data=80)
```

定义Model并进行模型训练，通过`LossMonitor`回调函数查看在训练过程中的损失函数值：

```python
from mindspore import Model
from mindspore.train.callback import LossMonitor

model = Model(network=net, loss_fn=crit, optimizer=opt, metrics=metrics)
# 获取训练过程数据
epochs = 2
model.train(epochs, train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

执行结果如下：

```text
epoch: 1 step: 1, loss is 158.6485
epoch: 1 step: 2, loss is 56.015274
epoch: 1 step: 3, loss is 22.507223
epoch: 1 step: 4, loss is 29.29523
epoch: 1 step: 5, loss is 54.613194
epoch: 1 step: 6, loss is 119.0715
epoch: 1 step: 7, loss is 47.707245
epoch: 1 step: 8, loss is 6.823062
epoch: 1 step: 9, loss is 12.838973
epoch: 1 step: 10, loss is 24.879482
epoch: 2 step: 1, loss is 38.01019
epoch: 2 step: 2, loss is 34.66765
epoch: 2 step: 3, loss is 13.370583
epoch: 2 step: 4, loss is 3.0936844
epoch: 2 step: 5, loss is 6.6003437
epoch: 2 step: 6, loss is 19.703354
epoch: 2 step: 7, loss is 28.276491
epoch: 2 step: 8, loss is 10.402792
epoch: 2 step: 9, loss is 6.908296
epoch: 2 step: 10, loss is 1.5971221
```

执行模型评估，获取评估结果：

```python
eval_result = model.eval(eval_dataset)
print(eval_result)
```

执行结果如下：

```text
{'mae': 2.4565244197845457}
```

使用`predict`进行推理：

```python
for d in eval_dataset.create_dict_iterator():
    data = d["data"]
    break

output = model.predict(data)
print(output)
```

执行结果如下：

```text
[[ 13.330149  ]
 [ -3.380001  ]
 [ 11.5734005 ]
 [ -0.84721684]
 [ 11.391014  ]
 [ -9.029837  ]
 [  1.1881653 ]
 [  2.1025467 ]
 [ 13.401606  ]
 [  1.8194647 ]
 [  8.862836  ]
 [ 14.427877  ]
 [  4.330497  ]
 [-12.431898  ]
 [ -4.5104184 ]
 [  9.439548  ]]
```

一般情况下需要对推理结果进行后处理才能得到比较直观的推理结果。

与构建网络后直接运行不同，使用Model进行模型训练、推理和评估时，不需要`set_train`配置网络结构的执行模式。

## 自定义场景的Model应用

在[损失函数](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/loss.html)和[构建训练与评估网络](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/train_and_eval.html)中已经提到过，MindSpore提供的网络封装函数`nn.WithLossCell`、`nn.TrainOneStepCell`和`nn.WithEvalCell`并不适用于所有场景，实际场景中常常需要自定义网络的封装方式。这种情况下`Model`使用这些封装函数自动地进行封装显然是不合理的。接下来介绍这些场景下如何正确地使用`Model`。

### 手动连接前向网络与损失函数

在有多个数据或者多个标签的场景下，可以手动将前向网络和自定义的损失函数链接起来作为`Model`的`network`，`loss_fn`使用默认值`None`，此时`Model`内部便会直接使用`nn.TrainOneStepCell`将`network`与`optimizer`组成训练网络，而不会经过`nn.WithLossCell`。

这里使用`损失函数`<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/loss.html>文档中的例子：

1. 定义多标签数据集

    ```python
    import numpy as np
    import mindspore.dataset as ds

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

2. 自定义多标签损失函数

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

3. 连接前向网络和损失函数，`net`使用上一节定义的`LinearNet`

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
    net = LinearNet()
    loss = L1LossForMultiLabel()
    loss_net = CustomWithLossCell(net, loss)
    ```

4. 定义Model并进行模型训练

    ```python
    from mindspore.train.callback import LossMonitor
    from mindspore import Model

    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    model = Model(network=loss_net, optimizer=opt)
    multi_train_dataset = create_multilabel_dataset(num_data=160)
    model.train(epoch=1, train_dataset=multi_train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)
    ```

    执行结果如下：

    ```text
    epoch: 1 step: 1, loss is 2.7395597
    epoch: 1 step: 2, loss is 3.730921
    epoch: 1 step: 3, loss is 6.393111
    epoch: 1 step: 4, loss is 5.684395
    epoch: 1 step: 5, loss is 6.089678
    epoch: 1 step: 6, loss is 8.953241
    epoch: 1 step: 7, loss is 9.357056
    epoch: 1 step: 8, loss is 8.601417
    epoch: 1 step: 9, loss is 9.339062
    epoch: 1 step: 10, loss is 7.6557174
    ```

5. 模型评估

    `Model`默认使用`nn.WithEvalCell`构建评估网络，在不满足需求的情况下同样需要手动构建评估网络，多数据和多标签便是一个典型的场景。`Model`提供了`eval_network`用于设置自定义的评估网络。手动构建评估网络的方式如下：

    自定义评估网络的封装方式：

    ```python
    import mindspore.nn as nn

    class CustomWithEvalCell(nn.Cell):
        def __init__(self, network):
            super(CustomWithEvalCell, self).__init__(auto_prefix=False)
            self.network = network

        def construct(self, data, label1, label2):
            output = self.network(data)
            return output, label1, label2
    ```

    手动构建评估网络：

    ```python
    eval_net = CustomWithEvalCell(net)
    ```

    使用Model进行模型评估：

    ```python
    from mindspore.train.callback import LossMonitor
    from mindspore import Model

    mae1 = nn.MAE()
    mae2 = nn.MAE()
    mae1.set_indexes([0, 1])
    mae2.set_indexes([0, 2])

    model = Model(network=loss_net, optimizer=opt, eval_network=eval_net, metrics={"mae1": mae1, "mae2": mae2})
    multi_eval_dataset = create_multilabel_dataset(num_data=80)
    result = model.eval(multi_eval_dataset, dataset_sink_mode=False)
    print(result)
    ```

    执行结果如下：

    ```text
    {'mae1': 8.572821712493896, 'mae2': 8.346409797668457}
    ```

    - 在进行模型评估时，评估网络的输出会透传给评估指标的`update`函数，也就是说，`update`函数将接收到三个输入，分别为`logits`、`label1`和`label2`。`nn.MAE`仅允许在两个输入上计算评价指标，因此使用`set_indexes`指定`mae1`使用下标为0和1的输入，也就是`logits`和`label1`，计算评估结果；指定`mae2`使用下标为0和2的输入，也就是`logits`和`label2`，计算评估结果。

    - 在实际场景中，往往需要所有标签同时参与评估，这时候就需要自定义`Metric`，灵活使用评估网络的所有输出计算评估结果。`Metric`自定义方法详见：<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/self_define_metric.html>。

6. 推理

   `Model`没有提供用于指定自定义推理网络的参数，此时可以直接运行前向网络获得推理结果。

    ```python
    for d in multi_eval_dataset.create_dict_iterator():
        data = d["data"]
        break

    output = net(data)
    print(output)
    ```

     执行结果如下：

    ```text
    [[ 7.147398  ]
    [ 3.4073524 ]
    [ 7.1618156 ]
    [ 1.8599509 ]
    [ 0.8132744 ]
    [ 4.92359   ]
    [ 0.6972816 ]
    [ 6.6525955 ]
    [ 1.2478441 ]
    [ 2.791972  ]
    [-1.2134678 ]
    [ 7.424588  ]
    [ 0.24634433]
    [ 7.15598   ]
    [ 0.68831706]
    [ 6.171982  ]]
    ```  

### 自定义训练网络

在自定义`TrainOneStepCell`时，需要手动构建训练网络作为`Model`的`network`，`loss_fn`和`optimizer`均使用默认值`None`，此时`Model`会使用`network`作为训练网络，而不会进行任何封装。

自定义`TrainOneStepCell`的场景可参考[构建训练与评估网络](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/train_and_eval.html)，这里列举一个简单的例子：

```python
from mindspore.nn import TrainOneStepCell as CustomTrainOneStepCell
from mindspore import Model
from mindspore.train.callback import LossMonitor

# 手动构建训练网络
train_net = CustomTrainOneStepCell(loss_net, opt)
# 定义`Model`并执行训练
model = Model(train_net)
multi_train_ds = create_multilabel_dataset(num_data=160)
model.train(epoch=1, train_dataset=multi_train_ds, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

执行结果如下：

```text
epoch: 1 step: 1, loss is 8.834492
epoch: 1 step: 2, loss is 9.452023
epoch: 1 step: 3, loss is 6.974942
epoch: 1 step: 4, loss is 5.8168106
epoch: 1 step: 5, loss is 5.6446257
epoch: 1 step: 6, loss is 4.7653127
epoch: 1 step: 7, loss is 4.059086
epoch: 1 step: 8, loss is 3.5931993
epoch: 1 step: 9, loss is 2.8107128
epoch: 1 step: 10, loss is 2.3682175
```

此时`train_net`即为训练网络。自定义训练网络时，同样需要自定义评估网络，进行模型评估和推理的方式与上一节`手动连接前向网络与损失函数`相同。

当自定义训练网络的标签和预测值均为单一值时，评价函数不需要特殊处理(自定义或使用`set_indexes`)，其他场景仍然需要注意评价指标的正确使用方式。

### 自定义网络的权重共享

[构建训练与评估网络](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/train_and_eval.html)中已经介绍过权重共享的机制，使用MindSpore构建不同网络结构时，只要这些网络结构是在同一个实例的基础上封装的，那这个实例中的所有权重便是共享的，一个网络结构中的权重发生变化，意味着其他网络结构中的权重同步发生了变化。

在使用Model进行训练时，对于简单的场景，`Model`内部使用`nn.WithLossCell`、`nn.TrainOneStepCell`和`nn.WithEvalCell`在前向`network`实例的基础上构建训练和评估网络，`Model`本身确保了推理、训练、评估网络之间权重共享。但对于自定义使用Model的场景，用户需要注意前向网络仅实例化一次。如果构建训练网络和评估网络时分别实例化前向网络，那在使用`eval`进行模型评估时，便需要手动加载训练网络中的权重，否则模型评估使用的将是初始的权重值。
