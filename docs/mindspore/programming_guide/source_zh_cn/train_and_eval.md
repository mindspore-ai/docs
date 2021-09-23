# 构建和执行网络模型

`Linux` `Ascend` `GPU` `CPU` `模型开发` `高级`

<!-- TOC -->

- [构建和执行网络模型](#构建和执行网络模型)
    - [概述](#概述)
    - [构建前向网络](#构建前向网络)
    - [构建训练网络](#构建训练网络)
        - [使用训练网络包装函数](#使用训练网络包装函数)
        - [创建数据集并执行训练](#创建数据集并执行训练)
        - [自定义训练网络包装函数](#自定义训练网络包装函数)
    - [构建评估网络](#构建评估网络)
        - [使用评估网络包装函数](#使用评估网络包装函数)
        - [创建数据集并执行评估](#创建数据集并执行评估)
        - [自定义评估网络包装函数](#自定义评估网络包装函数)
    - [构建网络的权重共享](#构建网络的权重共享)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/train_and_eval.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

前面章节讲解了MindSpore构建网络所使用的基本元素，如MindSpore的网络基本单元、损失函数、优化器等。本文档重点介绍如何使用这些元素组成训练和评估网络。

## 构建前向网络

使用Cell构建前向网络，这里定义一个简单的线性回归LinearNet：

> Cell的用法详见<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/build_net.html>

```python
import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)
```

## 构建训练网络

构建训练网络需要在前向网络的基础上叠加损失函数、反向传播和优化器。

### 使用训练网络包装函数

MindSpore的nn模块提供了训练网络封装函数`TrainOneStepCell`，下面使用`nn.TrainOneStepCell`将前面定义的LinearNet封装成一个训练网络。具体过程如下：

```python
# 实例化前向网络
net = LinearNet()
# 设定损失函数并连接前向网络与损失函数
crit = nn.MSELoss()
net_with_criterion = nn.WithLossCell(net, crit)
# 设定优化器
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# 定义训练网络
train_net = nn.TrainOneStepCell(net_with_criterion, opt)
# 设置网络为训练模式
train_net.set_train()
```

`set_train`递归地配置了`Cell`的`training`属性，在实现训练和推理结构不同的网络时可以通过`training`属性区分训练和推理场景，例如`BatchNorm`、`Dropout`。

前面的[损失函数](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/loss.html)章节已经介绍了如何定义损失函数，以及如何使用`WithLossCell`将前向网络与损失函数连接起来，这里介绍如何获取梯度和更新权重，构成一个完整的训练网络。MindSpore提供的`nn.TrainOneStepCell`具体实现如下：

```python
import mindspore.ops as ops
from mindspore.context import get_auto_parallel_context, ParallelMode
from mindspore.communication import get_group_size

def get_device_num():
    """Get the device num."""
    parallel_mode = auto_parallel_context().get_parallel_mode()
    if parallel_mode == "stand_alone":
        device_num = 1
        return device_num

    if auto_parallel_context().get_device_num_is_set() is False:
        device_num = get_group_size()
    else:
        device_num = auto_parallel_context().get_device_num()
    return device_num

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = ops.identity
        self.parallel_mode = auto_parallel_context().get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = auto_parallel_context().get_gradients_mean()
            self.degree = get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
```

`TrainOneStepCell`中包含入参：

- network (Cell)：参与训练的网络，该网络包含前向网络和损失函数的计算逻辑，输入数据和标签，输出损失函数值。

- optimizer (Cell)：所使用的优化器。

- sens (float)：反向传播缩放比例。

`TrainOneStepCell`初始化时还定义了以下内容：

- GradOperation：反向传播函数，用于进行反向传播并获取梯度。

- DistributedGradReducer：用于在分布式场景下进行梯度广播，单机单卡不需要使用。

`construct`定义的训练执行过程主要包含4个步骤：

- `loss = self.network(*inputs)`：执行前向网络，计算当前输入的损失函数值。
- `grads = self.grad(self.network, self.weights)(*inputs, sens)`：进行反向传播，计算梯度。
- `grads = self.grad_reducer(grads)`：在分布式情况下进行梯度广播，单机单卡时直接返回输入梯度。
- `self.optimizer(grads)`：使用优化器更新权重。

### 创建数据集并执行训练

生成数据集并进行数据预处理：

```python
import mindspore.dataset as ds
import numpy as np

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

train_dataset = create_dataset(num_data=160)
```

进行模型训练：

```python
# 获取训练过程数据
epochs = 2
for epoch in range(epochs):
    for d in train_dataset.create_dict_iterator():
        result = train_net(d["data"], d["label"])
        print(result)
```

使用`nn.TrainOneStepCell`封装的训练网络的输出损失函数值，执行结果如下：

```text
144.26233
81.79023
11.277914
29.376678
191.91623
92.79765
25.821865
7.4363556
41.602726
38.070984
51.20244
31.435104
8.940489
20.17907
58.80686
33.43603
12.905434
4.689845
18.978374
35.082695
```

### 自定义训练网络包装函数

一般情况下，用户可以使用框架提供的`nn.TrainOneStepCell`封装训练网络，在`nn.TrainOneStepCell`不能满足需求时，则需要自定义符合实际场景的`TrainOneStepCell`。例如：

1、ModelZoo中的Bert就在`nn.TrainOneStepCell`的基础上，加入了梯度截断操作，以获得更好的训练效果，Bert定义的训练包装函数代码片段如下：

> Bert网络详见：https://gitee.com/mindspore/models/tree/master/official/nlp/bert

```python
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = ops.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                   ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad

class BertTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(BertTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = ops.Cast()
        self.hyper_map = ops.HyperMap()
        self.enable_clip_grad = enable_clip_grad

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        grads = self.grad(self.network, weights)(*inputs, self.cast(ops.tuple_to_array((self.sens,)), mstype.float32))
        if self.enable_clip_grad:
            # 进行梯度截断
            grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
```

2、Wide&Deep输出两个损失函数值，并对网络的Wide和Deep两部分分别进行反向传播和参数更新，而`nn.TrainOneStep`仅适用于一个损失函数值的场景，因此ModelZoo中Wide&Deep自定义了训练封装函数，代码片段如下：

> Wide&Deep网络详见：https://gitee.com/mindspore/models/tree/master/official/recommend/wide_and_deep

```python
class IthOutputCell(nn.Cell):
    """
    IthOutputCell
    """
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, *inputs):
        """
        IthOutputCell construct
        """
        predict = self.network(*inputs)[self.output_index]
        return predict

class TrainStepWrap(nn.Cell):
    def __init__(self, network, config, sens=1000.0):
        super(TrainStepWrap, self).__init__()
        self.network = network
        self.network.set_train()
        self.trainable_params = network.trainable_params()
        weights_w = []
        weights_d = []
        for params in self.trainable_params:
            if 'wide' in params.name:
                weights_w.append(params)
            else:
                weights_d.append(params)

        self.weights_w = ParameterTuple(weights_w)
        self.weights_d = ParameterTuple(weights_d)
        self.optimizer_w = nn.FTRL(learning_rate=config.ftrl_lr,
                                   params=self.weights_w,
                                   l1=5e-4,
                                   l2=5e-4,
                                   initial_accum=0.1,
                                   loss_scale=sens)

        self.optimizer_d = nn.Adam(self.weights_d,
                                   learning_rate=config.adam_lr,
                                   eps=1e-6,
                                   loss_scale=sens)

        self.hyper_map = ops.HyperMap()

        self.grad_w = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_d = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)
        self.loss_net_w.set_grad()
        self.loss_net_w.set_grad()

        self.reducer_flag = False
        self.grad_reducer_w = None
        self.grad_reducer_d = None
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL,
                             ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer_w = DistributedGradReducer(
                self.optimizer_w.parameters, mean, degree)
            self.grad_reducer_d = DistributedGradReducer(
                self.optimizer_d.parameters, mean, degree)

    def construct(self, *inputs):
        """
        TrainStepWrap construct
        """
        weights_w = self.weights_w
        weights_d = self.weights_d
        loss_w, loss_d = self.network(*inputs)

        sens_w = ops.Fill()(ops.DType()(loss_w), ops.Shape()(loss_w), self.sens)
        sens_d = ops.Fill()(ops.DType()(loss_d), ops.Shape()(loss_d), self.sens)
        grads_w = self.grad_w(self.loss_net_w, weights_w)(*inputs, sens_w)
        grads_d = self.grad_d(self.loss_net_d, weights_d)(*inputs, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_w = self.grad_reducer_w(grads_w)
            grads_d = self.grad_reducer_d(grads_d)
        return ops.depend(loss_w, self.optimizer_w(grads_w)), ops.depend(
            loss_d, self.optimizer_d(grads_d))
```

## 构建评估网络

评估网络的功能是输出预测值和真实标签，以便在验证集上评估模型训练的效果。MindSpore同样提供了评估网络包装函数`nn.WithEvalCell`。

### 使用评估网络包装函数

使用前面定义的前向网络和损失函数构建一个评估网络，具体过程如下：

```python
# 构建评估网络
eval_net = nn.WithEvalCell(net, crit)
eval_net.set_train(False)
```

执行`eval_net`输出预测值和标签，并使用评估指标进行处理，便可获得模型评估结果。`nn.WithEvalCell`的具体定义如下：

```python
class WithEvalCell(nn.Cell):
    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, data, label):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label
```

`WithEvalCell`中包含入参：

- network (Cell)：前向网络，输入数据和标签，并输出预测值。

- loss_fn (Cell)：所使用的损失函数，MindSpore提供的`WithEvalCell`输出`loss`，以便于将损失函数也作为一个评价指标，实际场景中`loss`并不是必须的输出项。

- add_cast_fp32 (Bool)：是否使用float32精度计算损失函数，目前该参数仅在`Model`使用`nn.WithEvalCell`构建评估网络时生效。

`construct`定义的训练执行过程主要包含2个步骤：

- `outputs = self._network(data)`：执行前向网络，计算当前输入数据的预测值。

- `return loss, outputs, label`：输出当前输入的损失函数值、预测值和标签。

### 创建数据集并执行评估

定义模型评价指标：

```python
mae = nn.MAE()
loss = nn.Loss()
```

使用前面定义的`DatasetGenerator`创建验证集：

```python
eval_dataset = create_dataset(num_data=160)
```

遍历数据集，执行`eval_net`，并使用`eval_net`的输出计算评估指标：

```python
mae.clear()
loss.clear()
for d in eval_dataset.create_dict_iterator():
    outputs = eval_net(d["data"], d["label"])
    mae.update(outputs[1], outputs[2])
    loss.update(outputs[0])

mae_result = mae.eval()
loss_result = loss.eval()
print("mae: ", mae_result)
print("loss: ", loss_result)
```

执行结果如下：

```text
mae: 3.948960852622986
loss: 21.080975341796876
```

`nn.WithEvalCell`输出损失函数值以便于计算评价指标`Loss`，如不需要可忽略该输出。

由于数据和权重具有随机性，因此训练结果具有随机性。

### 自定义评估网络包装函数

前面我们讲解了`nn.WithEvalCell`的计算逻辑，注意到`nn.WithEvalCell`只有两个输入data和label，对于多个数据或多个标签的场景显然不适用，此时如果想要构建评估网络就需要自定义`WithEvalCell`。这是因为评估网络需要使用数据计算预测值，并输出标签，当用户向`WithEvalCell`传入大于两个的输入时，框架无法识别这些输入中哪些是数据，哪些是标签。在自定义时，如不需要损失函数作为评价指标，则无需定义`loss_fn`。

以输入三个输入`data`, `label1`, `label2`为例，可以采用如下方式自定义`WithEvalCell`:

```python
class CustomWithEvalCell(nn.Cell):
    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._network = network

    def construct(self, data, label1, label2):
        outputs = self._network(data)
        return outputs, label1, label2

eval_net = CustomWithEvalCell(net)
eval_net.set_train(False)
```

MindSpore提供的基础评估指标仅适用于两个输入logits和label，当评估网络输出多个标签或多个预测值时，需要调用set_indexes函数指定哪几个输出用于计算评价指标。如果多个输出均需要用于计算评价指标，意味着MindSpore内置的评价指标不能满足需求，需要自定义。

Metric的使用方法和自定义方式详见<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/self_define_metric.html>。

## 构建网络的权重共享

通过前面的介绍可以看出，前向网络、训练网络和评估网络具有不同的逻辑，因此在需要时我们会构建三张网络。我们经常使用训练好的模型进行推理和评估，这就需要推理和评估网络中的权重值与训练网络中相同。使用模型保存和加载接口，将训练好的模型保存下来，再加载到推理和评估网络中，可以确保权重值相同。在训练平台上完成模型训练，再到其他推理平台进行推理时，模型保存与加载是必不可少的。

但在网络调测过程中，或使用边训练边验证方式进行模型调优时，往往在同一Python脚本中完成模型训练，评估或推理，此时MindSpore的权重共享机制可确保不同网络间的权重一致性。

使用MindSpore构建不同网络结构时，只要这些网络结构是在一个实例的基础上封装的，那这个实例中的所有权重便是共享的，一个网络中的权重发生变化，意味着其他网络中的权重同步发生了变化。

在本文档中，定义训练和评估网络时便使用了权重共享机制：

```python
# 实例化前向网络
net = LinearNet()
# 设定损失函数并连接前向网络与损失函数
crit = nn.MSELoss()
net_with_criterion = nn.WithLossCell(net, crit)
# 设定优化器
opt = nn.Adam(params=net.trainable_params())
# 定义训练网络
train_net = nn.TrainOneStepCell(net_with_criterion, opt)
train_net.set_train()
# 构建评估网络
eval_net = nn.WithEvalCell(net, crit)
eval_net.set_train(False)
```

`train_net`和`eval_net`均在`net`实例的基础上封装，因此在进行模型评估时，不需要加载`train_net`的权重。

若在构建`eval_net`时重新的定义前向网络，那`train_net`和`eval_net`之间便没有共享权重，如下：

```python
# 实例化前向网络
net = LinearNet()
# 设定损失函数并连接前向网络与损失函数
crit = nn.MSELoss()
net_with_criterion = nn.WithLossCell(net, crit)
# 设定优化器
opt = nn.Adam(params=net.trainable_params())
# 定义训练网络
train_net = nn.TrainOneStepCell(net_with_criterion, opt)
train_net.set_train()

# 再次实例化前向网络
net2 = LinearNet()
# 构建评估网络
eval_net = nn.WithEvalCell(net2, crit)
eval_net.set_train(False)
```

此时，若要在模型训练后进行评估，就需要将`train_net`中的权重加载到`eval_net`中。在同一脚本中进行模型训练、评估和推理时，利用好权重共享机制不失为一种更简便的方式。需要注意的是，如果前向网络结构中构建了训练和推理两种场景，同样需要确保满足权重共享的条件，如果分支语句中使用了同样的权重，该权重相关的网络结构只实例化一次。

这里讲解了如何构建和执行网络模型，后续章节会进一步讲解如何通过高阶API`Model`进行模型训练和评估。
