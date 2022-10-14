# Basic Usage of Models

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/model/model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Generally, defining a training and evaluation network and running it directly can meet basic requirements.

On the one hand, `Model` can simplify code to some extent. For example, you do not need to manually traverse datasets. If you do not need to customize `nn.TrainOneStepCell`, you can use `Model` to automatically build a training network. You can use the `eval` API of `Model` to evaluate the model and directly output the evaluation result. You do not need to manually invoke the `clear`, `update`, and `eval` functions of evaluation metrics.

On the other hand, `Model` provides many high-level functions, such as data offloading and mixed precision. Without the help of `Model`, it takes a long time to customize these functions by referring to `Model`.

The following describes MindSpore models and how to use `Model` for model training, evaluation, and inference.

![model](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/advanced/model/images/model.png)

## Introduction to Model

[Model](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model#mindspore.train.Model) is a high-level API provided by MindSpore for model training, evaluation, and inference. The common parameters of the API are as follows:

- `network`: neural network used for training or inference.
- `loss_fn`: used loss function.
- `optimizer`: used optimizer.
- `metrics`: evaluation function used for model evaluation.
- `eval_network`: network used for model evaluation. If the network is not defined, `Model` uses `network` and `loss_fn` for encapsulation.

`Model` provides the following APIs for model training, evaluation, and inference:

- `train`: used for model training on the training set.
- `eval`: used to evaluate the model on the evaluation set.
- `predict`: performs inference on a group of input data and outputs the prediction result.

### Using the Model API

For a neural network in a simple scenario, you can specify the feedforward network `network`, loss function `loss_fn`, optimizer `optimizer`, and evaluation function `metrics` when defining `Model`.

In this case, `Model` uses `network` as the feedforward network, uses `nn.WithLossCell` and `nn.TrainOneStepCell` to build a training network, and uses `nn.WithEvalCell` to build an evaluation network.

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore as ms
from mindspore.train import Model
from mindspore.common.initializer import Normal

def get_data(num, w=2.0, b=3.0):
    """Generate sample data and corresponding labels."""
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16):
    """Generate a dataset."""
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset

class LinearNet(nn.Cell):
    """Define the linear regression network.""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

train_dataset = create_dataset(num_data=160)
net = LinearNet()
crit = nn.MSELoss()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# Use a model to build a training network.
model = Model(network=net, loss_fn=crit, optimizer=opt, metrics={"mae"})
```

### Model Training

Use the `train` API to perform model training. The common parameters of the `train` API are as follows:

- `epoch`: number of training epoch. Generally, each epoch uses the full dataset for training.
- `train_dataset`: Iterator of a training dataset.
- `callbacks`: callback object or callback object list to be executed during training.

Interestingly, if `loss_fn` is defined in the network model, the data and label are transferred to `network` and `loss_fn` respectively. In this case, a tuple (data, label) needs to be returned for the dataset. If a dataset contains multiple pieces of data or labels, you can set `loss_fn` to None and implement a customized loss function in the `network`. In this case, tuples (data1, data2, data3, ...) consisting of all data returned by the dataset are transferred to the `network`.

The following example uses the `train` API to perform model training and uses the `LossMonitor` callback function to view the loss function value during training.

```python
from mindvision.engine.callback import LossMonitor

# Model training. The input parameter 0.005 of LossMonitor indicates the learning rate.
model.train(1, train_dataset, callbacks=[LossMonitor(0.005)])
```

```text
    Epoch:[  0/  1], step:[    1/   10], loss:[115.354/115.354], time:242.467 ms, lr:0.00500
    Epoch:[  0/  1], step:[    2/   10], loss:[86.149/100.751], time:0.650 ms, lr:0.00500
    Epoch:[  0/  1], step:[    3/   10], loss:[17.299/72.934], time:0.712 ms, lr:0.00500
    Epoch:[  0/  1], step:[    4/   10], loss:[21.070/59.968], time:0.744 ms, lr:0.00500
    Epoch:[  0/  1], step:[    5/   10], loss:[42.781/56.530], time:0.645 ms, lr:0.00500
    Epoch:[  0/  1], step:[    6/   10], loss:[52.374/55.838], time:0.577 ms, lr:0.00500
    Epoch:[  0/  1], step:[    7/   10], loss:[53.629/55.522], time:0.588 ms, lr:0.00500
    Epoch:[  0/  1], step:[    8/   10], loss:[16.356/50.626], time:0.624 ms, lr:0.00500
    Epoch:[  0/  1], step:[    9/   10], loss:[5.504/45.613], time:0.730 ms, lr:0.00500
    Epoch:[  0/  1], step:[   10/   10], loss:[5.396/41.591], time:0.766 ms, lr:0.00500
    Epoch time: 259.696 ms, per step time: 25.970 ms, avg loss: 41.591
```

### Model Evaluation

The `eval` API is used for evaluation. The parameters of the `eval` API are as follows:

- `valid_dataset`: dataset of the evaluation model.
- `callbacks`: callback object or callback object list to be executed during evaluation.
- `dataset_sink_mode`: determines whether data is directly offloaded to the processor for processing.

```python
eval_dataset = create_dataset(num_data=80)  # Create an evaluation dataset.
eval_result = model.eval(eval_dataset)      # Execute model evaluation.
print(eval_result)
```

```text
    {'mae': 4.2325128555297855}
```

### Model Inference

The `predict` API is used for prediction. The parameters of the `predict` API are as follows:

- `predict_data`: prediction sample. The data can be a single tensor, tensor list, or tensor tuple.

```python
eval_data = eval_dataset.create_dict_iterator()
data = next(eval_data)
# Perform model prediction.
output = model.predict(data["data"])
print(output)
```

```text
    [[-6.9463778 ]
     [ 1.3816066 ]
     [13.233659  ]
     [11.863918  ]
     [ 0.73616135]
     [-0.1280173 ]
     [ 7.579297  ]
     [-4.9149694 ]
     [ 7.416003  ]
     [10.491856  ]
     [-5.7275047 ]
     [ 9.984399  ]
     [-7.156473  ]
     [ 2.7091386 ]
     [-6.3339615 ]
     [-6.0259247 ]]
```

Generally, you need to post-process the inference result to obtain an intuitive inference result.

## Customized Scenarios

The network encapsulation functions `nn.WithLossCell`, `nn.TrainOneStepCell`, and `nn.WithEvalCell` provided by MindSpore are not applicable to all scenarios. In actual scenarios, you need to customize network encapsulation functions. In this case, it is unreasonable for `Model` to use these encapsulation functions to automatically encapsulate packets.

Next, let's look at how to correctly use `Model` when customizing network encapsulation functions.

### Customized Loss Network

If there are multiple data records or labels, you can use a customized loss network to link the feedforward network and the customized loss function as the `network` of `Model`. The default value of `loss_fn` is `None`.

In this case, `Model` does not pass through `nn.WithLossCell`, `nn.TrainOneStepCell` is directly used to form a training network with the `optimizer`.

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore.train import Model
from mindspore.nn import LossBase
from mindvision.engine.callback import LossMonitor

def get_multilabel_data(num, w=2.0, b=3.0):
    """Generate multi-label data. A group of data x corresponds to two labels: y1 and y2."""
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise1 = np.random.normal(0, 1)
        noise2 = np.random.normal(-1, 1)
        y1 = x * w + b + noise1
        y2 = x * w + b + noise2
        yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)

def create_multilabel_dataset(num_data, batch_size=16):
    """Generate a multi-label dataset. One piece of data corresponds to two labels: label1 and label2."""
    dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
    dataset = dataset.batch(batch_size)
    return dataset

class L1LossForMultiLabel(LossBase):
    """Customize a multi-label loss function."""

    def __init__(self, reduction="mean"):
        super(L1LossForMultiLabel, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target1, target2):
        """There are three inputs: predicted value 'base', actual values 'target1' and 'target2'."""
        x1 = self.abs(base - target1)
        x2 = self.abs(base - target2)
        return self.get_loss(x1) / 2 + self.get_loss(x2) / 2

class CustomWithLossCell(nn.Cell):
    """Connect the feedforward network and loss function."""

    def __init__(self, backbone, loss_fn):
        """There are two inputs: feedforward network 'backbone' and loss function 'loss_fn'."""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2):
        output = self._backbone(data)                 # Network output obtained through forward computation
        return self._loss_fn(output, label1, label2)  # Obtain the multi-label loss value.

multi_train_dataset = create_multilabel_dataset(num_data=160)

# Build a linear regression network.
net = LinearNet()
# Multi-label loss function
loss = L1LossForMultiLabel()

# Connect linear regression networks and multi-label loss functions.
loss_net = CustomWithLossCell(net, loss)
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# Use the model to connect the network and optimizer. In this case, the model does not pass through nn.WithLossCell.
model = Model(network=loss_net, optimizer=opt)
# Use the train API for model training.
model.train(epoch=1, train_dataset=multi_train_dataset, callbacks=[LossMonitor(0.005)])
```

```text
    Epoch:[  0/  1], step:[    1/   10], loss:[11.036/11.036], time:212.864 ms, lr:0.00500
    Epoch:[  0/  1], step:[    2/   10], loss:[9.984/10.510], time:0.592 ms, lr:0.00500
    Epoch:[  0/  1], step:[    3/   10], loss:[9.300/10.107], time:0.660 ms, lr:0.00500
    Epoch:[  0/  1], step:[    4/   10], loss:[7.526/9.462], time:0.787 ms, lr:0.00500
    Epoch:[  0/  1], step:[    5/   10], loss:[6.959/8.961], time:0.715 ms, lr:0.00500
    Epoch:[  0/  1], step:[    6/   10], loss:[10.290/9.183], time:0.716 ms, lr:0.00500
    Epoch:[  0/  1], step:[    7/   10], loss:[10.067/9.309], time:0.770 ms, lr:0.00500
    Epoch:[  0/  1], step:[    8/   10], loss:[8.924/9.261], time:0.909 ms, lr:0.00500
    Epoch:[  0/  1], step:[    9/   10], loss:[7.257/9.038], time:0.884 ms, lr:0.00500
    Epoch:[  0/  1], step:[   10/   10], loss:[6.138/8.748], time:0.955 ms, lr:0.00500
    Epoch time: 232.046 ms, per step time: 23.205 ms, avg loss: 8.748
```

### Customized Training Network

When customizing a training network, you need to manually build a training network as the `network` of `Model`. The default values of `loss_fn` and `optimizer` are None. In this case, `Model` uses the `network` as the training network without any encapsulation.

The following example describes how to customize a training network `CustomTrainOneStepCell` and build a training network through the `Model` API.

```python
import mindspore.ops as ops
import mindspore as ms
from mindspore.train import Model
from mindvision.engine.callback import LossMonitor

class CustomTrainOneStepCell(nn.Cell):
    """Customize a training network."""

    def __init__(self, network, optimizer, sens=1.0):
        """There are three input parameters: training network, optimizer, and backward propagation scaling ratio."""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.optimizer = optimizer                #Define the optimizer.
        self.value_and_grad = ms.value_and_grad(network, grad_position=None, weights=network.trainable_params())

    def construct(self, *inputs):
        loss, grads = self.value_and_grad(*inputs) # Perform backward propagation and compute the gradient.
        self.optimizer(grads)                      # Use the optimizer to update gradients.
        return loss

multi_train_ds = create_multilabel_dataset(num_data=160)

# Manually build a training network.
train_net = CustomTrainOneStepCell(loss_net, opt)
# Build a training network.
model = Model(train_net)
# Perform model training.
model.train(epoch=1, train_dataset=multi_train_ds, callbacks=[LossMonitor(0.01)])
```

```text
    Epoch:[  0/  1], step:[    1/   10], loss:[5.165/5.165], time:183.006 ms, lr:0.01000
    Epoch:[  0/  1], step:[    2/   10], loss:[4.042/4.603], time:0.800 ms, lr:0.01000
    Epoch:[  0/  1], step:[    3/   10], loss:[3.385/4.197], time:0.886 ms, lr:0.01000
    Epoch:[  0/  1], step:[    4/   10], loss:[2.438/3.758], time:0.896 ms, lr:0.01000
    Epoch:[  0/  1], step:[    5/   10], loss:[2.457/3.498], time:0.819 ms, lr:0.01000
    Epoch:[  0/  1], step:[    6/   10], loss:[2.546/3.339], time:0.921 ms, lr:0.01000
    Epoch:[  0/  1], step:[    7/   10], loss:[4.569/3.515], time:0.973 ms, lr:0.01000
    Epoch:[  0/  1], step:[    8/   10], loss:[4.031/3.579], time:1.271 ms, lr:0.01000
    Epoch:[  0/  1], step:[    9/   10], loss:[6.138/3.864], time:1.035 ms, lr:0.01000
    Epoch:[  0/  1], step:[   10/   10], loss:[3.055/3.783], time:1.263 ms, lr:0.01000
    Epoch time: 203.473 ms, per step time: 20.347 ms, avg loss: 3.783
```

### Customized Evaluation Network

By default, the `Model` uses `nn.WithEvalCell` to build the evaluation network. If the requirements are not met, you need to manually build the evaluation network, for example, in the multi-data and multi-label scenarios.

The following example shows how to customize an evaluation network `CustomWithEvalCell` and use the `Model` API to build an evaluation network.

```python
import mindspore.nn as nn
import mindspore as ms
from mindspore.train import Model, MAE

class CustomWithEvalCell(nn.Cell):
    """Customize multi-label evaluation network."""

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label1, label2):
        output = self.network(data)
        return output, label1, label2

# Build a multi-label evaluation dataset.
multi_eval_dataset = create_multilabel_dataset(num_data=80)

# Build an evaluation network.
eval_net = CustomWithEvalCell(net)

# Evaluation function
mae1 = MAE()
mae2 = MAE()
mae1.set_indexes([0, 1])
mae2.set_indexes([0, 2])

# Use a model to build an evaluation network.
model = Model(network=loss_net, optimizer=opt, eval_network=eval_net,
                 metrics={"mae1": mae1, "mae2": mae2})
result = model.eval(multi_eval_dataset)
print(result)
```

```text
    {'mae1': 2.5686439752578734, 'mae2': 2.4921266555786135}
```

When the preceding code is used for model evaluation, the output of the evaluation network is transparently transmitted to the `update` function of the evaluation metric. The `update` function receives three inputs: `logits`, `label1`, and `label2`.

`nn.MAE` allows evaluation metrics to be computed only on two inputs. Therefore, `set_indexes` is used to specify `mae1` to use inputs whose subscripts are 0 and 1, that is, `logits` and `label1`, to compute the evaluation result. Specify `mae2` to use the inputs whose subscripts are 0 and 2, that is, `logits` and `label2`, to compute the evaluation result.

### Network Inference

   The `Model` does not provide parameters for specifying a customized inference network. In this case, you can directly run the feedforward network to obtain the inference result.

```python
for d in multi_eval_dataset.create_dict_iterator():
    data = d["data"]
    break

output = net(data)
print(output)
```

```text
    [[-21.598358 ]
     [ -1.0123782]
     [ 10.457726 ]
     [ 12.409237 ]
     [ 19.666183 ]
     [ -5.846529 ]
     [  9.387393 ]
     [  2.6558673]
     [-15.15129  ]
     [-14.876989 ]
     [ 19.112661 ]
     [ 22.647848 ]
     [  4.9035554]
     [ 20.119627 ]
     [ -8.339532 ]
     [ -2.7513359]]
```
