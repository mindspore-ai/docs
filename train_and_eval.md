# Construct training and evaluation network

`Ascend` `GPU` `CPU` `Model Development` `Model Serving` `Model Evaluation`

[![View source](https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/train_and_eval.md)

## Overview

The previous chapters explained the basic elements used by MindSpore to build the network, such as MindSpore's network basic unit, loss function and optimizer, etc. This document focuses on how to use these elements to train and evaluate networks.

## Build a feedforward network

You can use Cell to build a feedforward network, we construct a simple linear regression LinearNet here:

> For more details about Cell
<https://www.mindspore.cn/docs/programming_guide/en/master/build_net.html>

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

## Building a training network

Building a training network requires stacking a loss function, backpropagation, and an optimizer on top of the forward network.

### Wrapping function with training network

MindSpore's nn module provides a training network encapsulation function `TrainOneStepCell`. Next, we will encapsulate the previously defined LinearNet into a training network using `nn.TrainOneStepCell`. The specific process is as follows:

```python
# Instantiate the feedforward network
net = LinearNet()
# Set the loss function and connect the feedforward network with the loss function
crit = nn.MSELoss()
net_with_criterion = nn.WithLossCell(net, crit)
# Set optimizer
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# Define the training network
train_net = nn.TrainOneStepCell(net_with_criterion, opt)
# Set the network to training mode
train_net.set_train()
```

`set_train` recursively configures the `training` attribute of a `Cell`. When implementing networks with different training and inference structures, the training and inference scenarios can be distinguished by the `training` attribute, such as `BatchNorm`, `Dropout`.

The previous chapter [Loss Function](https://www.mindspore.cn/docs/programming_guide/en/master/loss.html) has introduced how to define the loss function and use `WithLossCell` to convert the feedforward network connected with the loss function, here is how to obtain gradients and update weights to form a complete training network. The specific implementation of `nn.TrainOneStepCell` provided by MindSpore is as follows:

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

`TrainOneStepCell` contains following input parameters:

- network (Cell): The network involved in training contains the computational logic of the feedforward network and the loss function, input data and labels, and output the loss function value.

- optimizer (Cell): The chosen optimizer.

- sens (float): Backpropagation scaling.

The following parameters will also be defined when `TrainOneStepCell` is initialized:

- GradOperation: Backpropagation function for backpropagating and obtaining gradients.

- DistributedGradReducer: It is used for gradient broadcasting in distributed computing scenarios, it does not need to be used for a stand-alone machine with a single graphic card.

The training execution process defined by `construct` includes 4 main steps:

- `loss = self.network(*inputs)`: Execute the feedforward network and calculate the loss function value of the current input.
- `grads = self.grad(self.network, self.weights)(*inputs, sens)`: Perform backpropagation and calculate the gradient.
- `grads = self.grad_reducer(grads)`: Gradient broadcasting is performed in a distributed situation, and the input gradient is directly returned when using a stand-alone machine with a single graphic card.
- `self.optimizer(grads)`: Use the optimizer to update the weights.

### Generate dataset and perform training action

Generate the dataset and perform data preprocessing:

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

Train the model using the the loss function output value of the training network encapsulated by `nn.TrainOneStepCell`:

```python
# Get training data generated during the process
epochs = 2
for epoch in range(epochs):
    for d in train_dataset.create_dict_iterator():
        result = train_net(d["data"], d["label"])
        print(result)
```

### Customize a network training wrapper function

In general, users can use the `nn.TrainOneStepCell` provided by the framework to encapsulate the training network. When the `nn.TrainOneStepCell` cannot meet the requirements, you need to customize the `TrainOneStepCell` that meets the actual situation. For example:

1. Based on `nn.TrainOneStepCell`, Bert in ModelZoo adds truncated gradient operation to achieve better training effect. The code snippet of the training wrapper function defined by Bert is as follows:

> For more details about Bert network : https://gitee.com/mindspore/models/tree/master/official/nlp/bert

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
            # perform gradient truncation
            grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
```

2. Wide&Deep outputs two loss function values, and performs back-propagation and parameter update for the Wide and Deep parts of the network respectively, while `nn.TrainOneStep` is only suitable to one loss function value, so Wide&Deep in ModelZoo has a customized trainning wrapper function, the code snippet is as follows:

> For more Wide&Deep network details: <https://gitee.com/mindspore/models/tree/master/official/recommend/wide_and_deep>.

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
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL,
                             ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = get_auto_parallel_context("gradients_mean")
            degree = get_auto_parallel_context("device_num")
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

## Build an evaluation network

The function of the evaluation network is to output predicted values and true labels in order to evaluate the effect of model training on the validation set. MindSpore also provides an evaluation network wrapper function `nn.WithEvalCell`.

### Use the evaluation network wrapper function

Build an evaluation network using the previously defined feedforward network and loss function:

```python
# Build an evaluation network
eval_net = nn.WithEvalCell(net, crit)
eval_net.set_train(False)
```

You can obtain the model evaluation results by executing `eval_net` to output predicted values and labels, processing them with evaluation metrics. The specific definition of `nn.WithEvalCell` is as follows:

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

`WithEvalCell` contains following input parameters:

- network (Cell): A feedforward network that inputs data and labels, and outputs predicted values.

- loss_fn (Cell): The loss function used. The `WithEvalCell` provided by MindSpore outputs `loss`, so that the loss function can also be used as an evaluation indicator. In actual scenarios, `loss` is not a necessary output item.

- add_cast_fp32 (Bool): Whether to use float32 precision to calculate the loss function, currently this parameter only takes effect when `Model` uses `nn.WithEvalCell` to build an evaluation network.

The training execution process defined by `construct` includes 2 main steps:

- `outputs = self._network(data)`: Executes a feedforward network to compute predictions for the current input data.

- `return loss, outputs, label`: Output the loss function value, predicted value and label of the current input.

### Create dataset and perform evaluation

Define model evaluation metrics:

```python
mae = nn.MAE()
loss = nn.Loss()
```

Create a validation set using the `DatasetGenerator` defined earlier:

```python
eval_dataset = create_dataset(num_data=160)
```

Iterate through the dataset, execute `eval_net`, and use the output of `eval_net` to calculate evaluation metrics:

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

`nn.WithEvalCell` outputs the value of the loss function to facilitate the calculation of the evaluation index `Loss`, if not needed, this output can be ignored.

Since the data and weights are random, the training results are also random.

### Customize evaluation network wrapper function

Earlier we explained the computation logic of `nn.WithEvalCell`, and noticed that `nn.WithEvalCell` only has two inputs data and label, which are obviously not applicable when there are multiple data or labels. In this case, if you want to build an evaluation network, you need to customize `WithEvalCell`. This is because the evaluation network needs to use the data to calculate predictions and output labels. When the user passes more than two inputs to `WithEvalCell`, the framework cannot identify which of these inputs are data and which are labels. There is no need to define `loss_fn` if the loss function is not required as an evaluation index while customizing.
Taking the input of three inputs `data`, `label1`, `label2` as an example, you can customize `WithEvalCell`:

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

The basic evaluation index provided by MindSpore is only applicable to two inputs logits and label. When the evaluation network outputs multiple labels or multiple predicted values, you need to call the set_indexes function to specify which outputs are used to calculate the evaluation index. If multiple outputs need to be used to calculate evaluation indicators, it means that the built-in evaluation indicators in MindSpore cannot meet the requirements and need to be customized.

Please refer to <<https://www.mindspore.cn/docs/programming_guide/en/master/self_define_metric.html>> for the usage and customization methods of Metric.

## Build the weight sharing of the network

As can be seen from the previous introduction, the feedforward network, training network and evaluation network have different logics, so we need to build three networks when needed. We often use the trained model for inference and evaluation, which requires the same weight values in the inference and evaluation network as in the training network. Use the model save and load interface to save the trained model and load it into the inference and evaluation network to ensure the same weight values. When model training is completed on the training platform, and then inference is performed on other inference platforms, model saving and loading are essential.

However, in the process of network debugging, or using the training-while-verification method for model tuning, model training, evaluation or inference are often completed in the same Python script. At this time, MindSpore's weight sharing mechanism can ensure that the weights between different networks are consistent.

When using MindSpore to build different network structures, as long as these network structures are encapsulated on the basis of an instance, all the weights in this instance are shared, when the weights in a single network change, weights in other networks will be changed to are synchronized.

In this document, the weight sharing mechanism is used when defining the training and evaluation network:

```python
# Instantiate the feedforward network
net = LinearNet()
# Set the loss function and connect to the feedforward network
crit = nn.MSELoss()
net_with_criterion = nn.WithLossCell(net, crit)
# Set optimizer
opt = nn. Adam(params=net.trainable_params())
# Define the training network
train_net = nn.TrainOneStepCell(net_with_criterion, opt)
train_net.set_train()
# Build the evaluation network
eval_net = nn.WithEvalCell(net, crit)
eval_net.set_train(False)
```

Both `train_net` and `eval_net` are encapsulated basded on `net` instance, so during model evaluation the weights of `train_net` do not need to be loaded.

If the feedforward network is re-defined when `eval_net` is constructed, there will be no shared weights between `train_net` and `eval_net`:

```python
# Instantiate the feedforward network
net = LinearNet()
# Set the loss function and connect to the feedforward network
crit = nn.MSELoss()
net_with_criterion = nn.WithLossCell(net, crit)
# Set optimizer
opt = nn. Adam(params=net.trainable_params())
# Define the training network
train_net = nn.TrainOneStepCell(net_with_criterion, opt)
train_net.set_train()

# Instantiate the feedforward network again
net2 = LinearNet()
# Build the evaluation network
eval_net = nn.WithEvalCell(net2, crit)
eval_net.set_train(False)
```

Till this point, if you need to evaluate the model after training, the weights from `train_net` need to be loaded into `eval_net`. Taking advantage of the weight sharing mechanism is a shortcut to perform model training, evaluation, and inference in the same script. It should be noted that if two scenarios of training and inference are constructed in the feedforward network structure, it is also necessary to ensure that the conditions of weight sharing are met. If the same weight is used in the branch statement, the network structure related to the weight should only be instantiated once.

This section explains how to build and execute network models, later chapters will further explain how to train and evaluate models through the high-level API `Model`.