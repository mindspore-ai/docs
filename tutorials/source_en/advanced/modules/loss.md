# Loss Function

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/advanced/modules/loss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

A loss function is also called objective function and is used to measure the difference between a predicted value and an actual value.

In deep learning, model training is a process of reducing the loss function value through continuous iteration. Therefore, it is very important to select a loss function in a model training process, and a good loss function can effectively improve model performance.

The `mindspore.nn` module provides many [general loss functions](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.nn.html#loss-function), but these functions cannot meet all requirements. In many cases, you need to customize the required loss functions. The following describes how to customize loss functions.

![lossfun.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/modules/images/loss_function.png)

## Built-in Loss Functions

The following introduces [loss functions](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.nn.html#loss-function) built in the `mindspore.nn` module.

For example, use `nn.L1Loss` to compute the mean absolute error between the predicted value and the target value.

$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|$$

N is the value of `batch_size` in the dataset.

$$\ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}$$

A value of the `reduction` parameter in `nn.L1Loss` may be `mean`, `sum`, or `none`. If `reduction` is set to `mean` or `sum`, a scalar tensor (dimension reduced) after mean or sum is output. If `reduction` is set to `none`, the shape of the output tensor is the broadcast shape.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

# Output a mean loss value.
loss = nn.L1Loss()
# Output a sum loss value.
loss_sum = nn.L1Loss(reduction='sum')
# Output the original loss value.
loss_none = nn.L1Loss(reduction='none')

input_data = ms.Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = ms.Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))

print("loss:", loss(input_data, target_data))
print("loss_sum:", loss_sum(input_data, target_data))
print("loss_none:\n", loss_none(input_data, target_data))
```

```text
    loss: 1.5
    loss_sum: 9.0
    loss_none:
     [[1. 0. 2.]
     [1. 2. 3.]]
```

## Customized Loss Functions

You can customize a loss function by defining the loss function based on either `nn.Cell` or `nn.LossBase`. `nn.LossBase` is inherited from `nn.Cell` and provides the `get_loss` method. The `reduction` parameter is used to obtain a sum or mean loss value and output a scalar.

The following describes how to define the mean absolute error (MAE) function by inheriting `Cell` and `LossBase`. The formula of the MAE algorithm is as follows:

$$ loss= \frac{1}{m}\sum_{i=1}^m\lvert y_i-f(x_i) \rvert$$

In the preceding formula, $f(x)$ indicates the predicted value, $y$ indicates the actual value of the sample, and $loss$ indicates the mean distance between the predicted value and the actual value.

### `nn.Cell`-based Loss Function Build

`nn.Cell` is the base class of MindSpore. It can be used to build networks and define loss functions. The process of defining a loss function using `nn.Cell` is similar to that of defining a common network. The difference is that the execution logic is to compute the error between the feedforward network output and the actual value.

The following describes how to customize the loss function `MAELoss` based on `nn.Cell`.

```python
import mindspore.ops as ops

class MAELoss(nn.Cell):
    """Customize the loss function MAELoss."""

    def __init__(self):
        """Initialize."""
        super(MAELoss, self).__init__()
        self.abs = ops.Abs()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, base, target):
        """Call the operator."""
        x = self.abs(base - target)
        return self.reduce_mean(x)

loss = MAELoss()

input_data = ms.Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32)) # Generate a predicted value.
target_data = ms.Tensor(np.array([0.1, 0.2, 0.2]).astype(np.float32)) # Generate the actual value.

output = loss(input_data, target_data)
print(output)
```

```text
    0.033333335
```

### `nn.LossBase`-based Loss Function Build

The process of building the loss function `MAELoss` based on [nn.LossBase](https://www.mindspore.cn/docs/en/r1.9/api_python/nn/mindspore.nn.LossBase.html#mindspore.nn.LossBase) is similar to that of building the loss function based on `nn.Cell`. The `__init__` and `construct` methods need to be rewritten.

`nn.LossBase` can use the `get_loss` method to apply `reduction` to loss computation.

```python
class MAELoss(nn.LossBase):
    """Customize the loss function MAELoss."""

    def __init__(self, reduction="mean"):
        """Initialize and compute the mean loss value."""
        super(MAELoss, self).__init__(reduction)
        self.abs = ops.Abs() # Compute the absolute value.

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x) # Return the mean loss value.

loss = MAELoss()

input_data = ms.Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32)) # Generate a predicted value.
target_data = ms.Tensor(np.array([0.1, 0.2, 0.2]).astype(np.float32)) # Generate the actual value.

output = loss(input_data, target_data)
print(output)
```

```text
    0.033333335
```

## Loss Function and Model Training

After the loss function `MAELoss` is customized, you can use the `train` API in the [Model](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.Model.html#mindspore.Model) API of MindSpore to train a model. When building a model, you need to transfer the feedforward network, loss function, and optimizer. The `Model` associates them internally to generate a network model that can be used for training.

In `Model`, the feedforward network and loss function are associated through [nn.WithLossCell](https://www.mindspore.cn/docs/en/r1.9/api_python/nn/mindspore.nn.WithLossCell.html#mindspore.nn.WithLossCell). `nn.WithLossCell` supports two inputs: `data` and `label`.

```python
import mindspore as ms
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import LossMonitor

def get_data(num, w=2.0, b=3.0):
    """Generate data and corresponding labels."""
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16):
    """Load the dataset."""
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset

class LinearNet(nn.Cell):
    """Define the linear regression network."""
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

ds_train = create_dataset(num_data=160)
net = LinearNet()
loss = MAELoss()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# Use the model API to associate the network, loss function, and optimizer.
model = ms.Model(net, loss, opt)
model.train(epoch=5, train_dataset=ds_train, callbacks=[LossMonitor()])
```

```text
epoch: 1 step: 10, loss is 7.047606468200684
epoch: 2 step: 10, loss is 2.6299655437469482
epoch: 3 step: 10, loss is 3.130779266357422
epoch: 4 step: 10, loss is 2.1852657794952393
epoch: 5 step: 10, loss is 1.413834571838379
```

## Multi-label Loss Function and Model Training

A simple mean absolute error loss function `MAELoss` is defined above. However, datasets of many deep learning applications are relatively complex. For example, data of an object detection network Faster R-CNN includes a plurality of labels, instead of simply one piece of data corresponding to one label. In this case, the definition and usage of the loss function are slightly different.

The following describes how to define a multi-label loss function in a multi-label dataset scenario and use a model for model training.

### Multi-label Dataset

In the following example, two groups of linear data $y1$ and $y2$ are fitted by using the `get_multilabel_data` function. The fitting target function is:

$$f(x)=2x+3$$

The final dataset should be randomly distributed around the function. The dataset is generated according to the following formula, where `noise` is a random value that complies with the standard normal distribution. The `get_multilabel_data` function returns data $x$, $y1$, and $y2$.

$$f(x)=2x+3+noise$$

Use `create_multilabel_dataset` to generate a multi-label dataset and set `column_names` in `GeneratorDataset` to ['data', 'label1', 'label2']. The returned dataset is in the format that one piece of `data` corresponds to two labels `label1` and `label2`.

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
    dataset = dataset.batch(batch_size) # Each batch has 16 pieces of data.
    return dataset
```

### Multi-label Loss Function

Define the multi-label loss function `MAELossForMultiLabel` for the multi-label dataset created in the previous step.

$$ loss1= \frac{1}{m}\sum_{i=1}^m\lvert y1_i-f(x_i) \rvert$$

$$ loss2= \frac{1}{m}\sum_{i=1}^m\lvert y2_i-f(x_i) \rvert$$

$$ loss = \frac{(loss1 + loss2)}{2}$$

In the preceding formula, $f(x)$ is the predicted value of the sample label, $y1$ and $y2$ are the actual values of the sample label, and $loss1$ is the mean distance between the predicted value and the actual value $y1$, $loss2$ is the mean distance between the predicted value and the actual value $y2$, and $loss$ is the mean value of the loss value $loss1$ and the loss value $loss2$.

The `construct` method in `MAELossForMultiLabel` has three inputs: predicted value `base`, actual values `target1` and `target2`. In `construct`, compute the errors between the predicted value and the actual value `target1` and between the predicted value and the actual value `target2`, the mean value of the two errors is used as the final loss function value.

The sample code is as follows:

```python
class MAELossForMultiLabel(nn.LossBase):
    def __init__(self, reduction="mean"):
        super(MAELossForMultiLabel, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, target1, target2):
        x1 = self.abs(base - target1)
        x2 = self.abs(base - target2)
        return (self.get_loss(x1) + self.get_loss(x2)) / 2
```

### Multi-label Model Training

When a `Model` is used to associate a specified feedforward network, loss function, and optimizer, `nn.WithLossCell` used in the `Model` by default accepts only two inputs: `data` and `label`. Therefore, it is not applicable to multi-label scenarios.

In the multi-label scenario, if you want to use a `Model` for model training, you need to associate the feedforward network with the multi-label loss function in advance, that is, customize the loss network.

- Define a loss network.

    The following example shows how to define the loss network `CustomWithLossCell`. The `backbone` and `loss_fn` parameters of the `__init__` method indicate the feedforward network and loss function, respectively. The input of the `construct` method is the sample input `data` and the sample actual labels `label1` and `label2`, respectively. Transfer the sample input `data` to the feedforward network `backbone`, and transfer the predicted value and two label values to the loss function `loss_fn`.

    ```python
    class CustomWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(CustomWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label1, label2):
            output = self._backbone(data)
            return self._loss_fn(output, label1, label2)
    ```

- Define and train the network model.

    When `Model` is used to connect the feedforward network, multi-label loss function, and optimizer, the `network` of `Model` is specified as the customized loss network `loss_net`, the loss function `loss_fn` is not specified, and the optimizer is still `Momentum`.

    If `loss_fn` is not specified, the `Model` considers that the logic of the loss function has been implemented in the `network` by default, and does not use `nn.WithLossCell` to associate the feedforward network with the loss function.

    ```python
    ds_train = create_multilabel_dataset(num_data=160)
    net = LinearNet()

    # Define a multi-label loss function.
    loss = MAELossForMultiLabel()

    # Define the loss network. Connect the feedforward network and multi-label loss function.
    loss_net = CustomWithLossCell(net, loss)

    # Define the optimizer.
    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

    # Define a Model. In the multi-label scenario, the loss function does not need to be specified for the Model.
    model = ms.Model(network=loss_net, optimizer=opt)

    model.train(epoch=5, train_dataset=ds_train, callbacks=[LossMonitor()])
    ```

    ```text
    epoch: 1 step: 10, loss is 7.504277229309082
    epoch: 2 step: 10, loss is 3.1470584869384766
    epoch: 3 step: 10, loss is 2.810225486755371
    epoch: 4 step: 10, loss is 2.774254322052002
    epoch: 5 step: 10, loss is 1.572474718093872
    ```

The preceding describes how to define a loss function and use a Model for model training in the multi-label dataset scenario. In many other scenarios, this method may also be used for model training.
