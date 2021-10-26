# 低阶API的使用

为方便用户控制整网的执行流程，MindSpore提供了高阶的训练和推理接口`mindspore.Model`，通过指定要训练的神经网络模型和常见的训练设置，调用`train`和`eval`方法对网络进行训练和推理。同时，用户如果想要对特定模块进行个性化设置，也可以调用对应的中低阶接口自行定义，本文介绍了如何使用中低阶API定义各个模块。

## 定义数据集

MindSpore的`mindspore.dataset`模块集成了常见的数据处理功能：用户既可以调用此模块的相关接口来[加载常见的数据集](https://www.mindspore.cn/tutorials/zh-CN/master/dataset.html#%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86)，也可以构造数据集类并配合使用`GeneratorDataset`接口，实现自定义数据集和数据加载。使用`GeneratorDataset`生成具有多项式关系的样本和对应的结果，代码样例如下：

导入所需的包：

```python
import numpy as np
import mindspore as ms
from mindspore import ops, nn
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.common.initializer as init
```

定义数据集：

```python
def get_data(data_num, data_size):
    for _ in range(data_num):
        data = np.random.randn(data_size)
        p = np.array([1, 0, -3, 5])
        label = np.polyval(p, data).sum()
        yield data.astype(np.float32), np.array([label]).astype(np.float32)

def create_dataset(data_num, data_size, batch_size=32, repeat_size=1):
    """定义数据集"""
    input_data = ds.GeneratorDataset(list(get_data(data_num, data_size)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data
```

## 定义网络

MindSpore的Cell类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，可以继承Cell类，并重写`__init__`方法和`construct`方法。MindSpore的`ops`模块提供了基础算子的实现，`nn`模块实现了对基础算子的进一步封装，用户可以根据需要，灵活使用不同的算子。

使用常用的`nn`算子构建一个简单的全连接网络：

```python
class MyNet(nn.Cell):
    """定义网络"""
    def __init__(self, input_size=32):
        super(MyNet, self).__init__()
        self.fc1 = nn.Dense(input_size, 120, weight_init=init.Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=init.Normal(0.02))
        self.fc3 = nn.Dense(84, 1, weight_init=init.Normal(0.02))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 定义损失函数

损失函数用于衡量预测值与真实值差异的程度。深度学习中，模型训练就是通过不停地迭代来缩小损失函数值的过程。因此在模型训练过程中损失函数的选择非常重要，定义一个好的损失函数，可以有效提高模型的性能。

MindSpore提供了许多通用损失函数供用户选择，也支持用户根据需要自定义损失函数。定义损失函数类时，既可以继承网络的基类`nn.Cell`，也可以继承损失函数的基类`nn.LossBase`。

### 继承Cell定义损失函数

`Cell`是MindSpore的基本网络单元，可以用于构建网络，损失函数也可以通过`Cell`来定义。使用`Cell`定义损失函数的方法与定义一个普通的网络的差别在于，其执行逻辑用于计算前向网络输出与真实值之间的误差。

以平均绝对误差为例，损失函数的定义方法如下：

```python
class MyL1Loss(nn.Cell):
    """定义损失"""
    def __init__(self):
        super(MyL1Loss, self).__init__()
        self.abs = ops.Abs()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, predict, target):
        x = self.abs(predict - target)
        return self.reduce_mean(x)
```

### 继承基类定义损失函数

在定义损失函数时还可以继承损失函数的基类`nn.LossBase`。损失函数的基类在`Cell`的基础上，提供了`get_loss`方法，用于对损失值求和或求均值，输出一个标量。MyL1Loss使用`LossBase`作为基类的定义如下：

```python
class MyL1Loss(nn.LossBase):
    """定义损失"""
    def __init__(self, reduction="mean"):
        super(MyL1Loss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x)
```

## 定义优化器

优化器在模型训练过程中，用于计算和更新网络参数，合适的优化器可以有效减少训练时间，提高最终模型性能。

MindSpore提供了许多通用的优化器供用户选择，同时也支持用户根据需要自定义优化器。自定义优化器时可以继承优化器基类`optimizer`，重写`construct`函数实现参数的更新。

使用基础的运算算子自定义优化器：

```python
class MyMomentum(nn.Optimizer):
    """定义优化器"""
    def __init__(self, params, learning_rate, momentum=0.9, use_nesterov=False):
        super(MyMomentum, self).__init__(learning_rate, params)
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.use_nesterov = use_nesterov
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.assign = ops.Assign()

    def construct(self, gradients):
        lr = self.get_lr()
        params = self.parameters
        for i in range(len(params)):
            self.assign(self.moments[i], self.moments[i] * self.momentum + gradients[i])
            if self.use_nesterov:
                update = params[i] - (self.moments[i] * self.momentum + gradients[i]) * lr
            else:
                update = params[i] - self.moments[i] * lr
            self.assign(params[i], update)
        return params
```

MindSpore也封装了`ApplyMomentum`算子供用户使用，使用`ApplyMomentum`算子自定义优化器：

```python
class MyMomentum(nn.Optimizer):
    """定义优化器"""
    def __init__(self, params, learning_rate, momentum=0.9, use_nesterov=False):
        super(MyMomentum, self).__init__(learning_rate, params)
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.momentum = momentum
        self.opt = ops.ApplyMomentum(use_nesterov=use_nesterov)

    def construct(self, gradients):
        params = self.parameters
        success = None
        for param, mom, grad in zip(params, self.moments, gradients):
            success = self.opt(param, mom, self.learning_rate, grad, self.momentum)
        return success
```

## 定义训练流程

MindSpore的nn模块提供了部分封装接口方便用户在训练过程中使用，例如，`nn.WithLossCell`接口可以将前向网络与损失函数连接起来；`nn.TrainOneStepCell`封装了损失网络和优化器，在执行训练时通过`GradOperation`算子来进行梯度的获取，通过优化器来实现权重的更新。

定义损失网络：

```python
class MyWithLossCell(nn.Cell):
    """定义损失网络"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        return self.backbone
```

MindSpore的nn模块提供了训练网络封装函数`nn.TrainOneStepCell`，如无特殊需求，用户可以继承`nn.TrainOneStepCell`来定义自己的训练流程。

```python
class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)
```

执行训练：

```python
# 生成多项式分布的训练数据
dataset_size = 32
ds_train = create_dataset(2048, dataset_size)
# 网络
net = MyNet()
# 损失函数
loss_func = MyL1Loss()
# 优化器
opt = MyMomentum(net.trainable_params(), 0.01)
# 构建损失网络
net_with_criterion = MyWithLossCell(net, loss_func)
# 构建训练网络
train_net = MyTrainStep(net_with_criterion, opt)
# 执行训练，每个epoch打印一次损失值
epochs = 5
for epoch in range(epochs):
    for train_x, train_y in ds_train:
        train_net(train_x, train_y)
        loss_val = net_with_criterion(train_x, train_y)
    print(loss_val)
```

输出：

```text
135.42422
32.48938
16.423292
14.1008625
11.454975
```

## 定义metric

当训练任务结束，常常需要评估函数来评估模型的好坏，MindSpore的`metric`模块提供了常见的评估函数，用户也可以根据需要自行定义评估指标：

```python
class MyMAE(nn.Metric):
    """定义metric"""
    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        self.abs_error_sum = 0
        self.samples_num = 0

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        error_abs = np.abs(y.reshape(y_pred.shape) - y_pred)
        self.abs_error_sum += error_abs.sum()
        self.samples_num += y.shape[0]

    def eval(self):
        return self.abs_error_sum / self.samples_num
```

## 定义验证流程

MindSpore的nn模块提供了评估网络包装函数`nn.WithEvalCell`，用户也可以自己定义评估网络包装函数：

```python
class MyWithEvalCell(nn.Cell):
    """定义验证流程"""
    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        outputs = self.network(data)
        return outputs, label
```

执行推理并评估：

```python
# 获取验证数据
ds_eval = create_dataset(128, dataset_size, 1)
# 定义评估网络
eval_net = MyWithEvalCell(net)
eval_net.set_train(False)
# 定义评估指标
mae = MyMAE()
# 执行推理过程
for eval_x, eval_y in ds_eval:
    output, eval_y = eval_net(eval_x, eval_y)
    mae.update(output, eval_y)

mae_result = mae.eval()
print("mae: ", mae_result)
```

输出评估误差，MAE与模型在训练集上效果大致相同：

```text
mae:  11.72380930185318
```

## 保存模型

使用MindSpore提供的`save_checkpoint`保存模型，传入网络和保存路径：

```python
ms.save_checkpoint(net, "./MyNet.ckpt")
```
