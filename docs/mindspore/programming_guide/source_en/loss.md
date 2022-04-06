# Loss Function

`Ascend` `GPU` `CPU` `Model Development`

Translator: [Misaka19998](https://gitee.com/Misaka19998)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/loss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Loss function, also known as object function, is used for measuring the difference between predicted and true value. In deep learning, training a model is a process of decrease the loss value by iteration. So it is important to choose a loss function while training a model. A better loss function can efficiently increase model's performance.

MindSpore provides many general loss functions for users. However, they are not suitable for all the situations. Users need to define their own loss functions in some cases. So this course will introduce how to define loss functions.

Currently, MindSpore supports the following loss functions: `L1Loss`, `MSELoss`, `SmoothL1Loss`, `SoftmaxCrossEntropyWithLogits`, `SampledSoftmaxLoss`, `BCELoss`, and `CosineEmbeddingLoss`.

All loss functions of MindSpore are implemented by subclasses of `Cell`. Therefore, customized loss functions are also supported. For details about how to build a loss function, see "Building a Customized Network."

### Built-in Loss Functions

- L1Loss

  Computes the absolute value error of two input data for the regression model. The default value of `reduction` is mean. If the value of `reduction` is sum, the loss accumulation result is returned. If the value of `reduction` is none, the result of each loss is returned.

- MSELoss

  Computes the square error of two input data for the regression model. The `reduction` parameter is the same as the `L1Loss` parameter.

- SmoothL1Loss

  `SmoothL1Loss` is the smooth L1 loss function, which is used for the regression model. The default value of the `beta` threshold is 1.

- SoftmaxCrossEntropyWithLogits

  Cross entropy loss function, which is used to classify models. If the tag data is not encoded in the one-hot mode, set `sparse` to True. The default value of `reduction` is none. The meaning of this parameter is the same as that of `L1Loss`.

- CosineEmbeddingLoss

  `CosineEmbeddingLoss` is used to measure the similarity between two inputs and is used for classification models. The default value of `margin` is 0.0. The `reduction` parameter is the same as the `L1Loss` parameter.

- BCELoss

  Binary cross entropy loss is used for binary classification. `weight` is a rescaling weight applied to the loss of each batch element. The default value of `weight` is None, which means the weight values are all 1. The default value of `reduction` parameter is none. The `reduction` parameter is the same as the `L1Loss` parameter.

- SampledSoftmaxLoss

  Sampled softmax loss function, which is used for classification model when the number of class is large. `num_sampled` is the number of classes to randomly sample. `num_class` is the number of possible classes. `num_true` is the number of target classes per training example. `sampled_values` is the sampled candidate. The default value of `sampled_values` is None, which means UniformCandidateSampler is applied. `remove_accidental_hits` is the switch of whether to remove "accidental hits". The default value of `remove_accidental_hits` is True. `seed` is the random seed for candidate sampling with the default value of 0. The default value of reduction parameter is none. The `reduction` parameter is the same as the L1Loss parameter.

### Built-in Loss Functions Application Cases

All loss functions of MindSpore are stored in mindspore.nn. The usage method is as follows:

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

In this case, two pieces of tensor data are built. The `nn.L1Loss` API is used to define the loss, `input_data` and `target_data` are transferred to the loss, and the L1Loss computation is performed. The result is 1.5. If loss is set to nn.L1Loss(reduction='sum'), the result is 9.0. If loss is set to nn.L1Loss(reduction='none'), the result is [[1. 0. 2.] [1. 2. 3.]].

## Defining Loss Function

Cell is the basic network module of MindSpore, and can be used to construct the network and define loss functions. The way to define a loss function is the same as defining a network. The difference is that its execution logic is used to calculate the error between the output of the forward network and the true value.

Taking a MindSpore loss function, L1 Loss, as an example. The way to define the loss function is as follow:

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

The needed operator will be instantiated in `__init__`method and used in `construct`. Then an L1Loss function is defined.

With a series of given predicted and true value, users can call the loss function to get the difference of them, as follow:

```python
import numpy as np
from mindspore import Tensor

loss = L1Loss()
input_data = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
target_data = Tensor(np.array([0.1, 0.2, 0.2]).astype(np.float32))

output = loss(input_data, target_data)
print(output)
```

Taking `Ascend` backup as an example, the output is as follow:

```text
0.03333334
```

When the loss function is defined, the base class `Loss` of the loss function can also be inherited. `Loss` provides the `get_loss` method, which is used to sum or average the loss values and output a scalar. The definition of L1Loss using `Loss` as the base class is as follows:

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

Firstly, we use `Loss` as the base class of L1Loss, and then add a parameter `reduction` to `__init__`, and then pass to base class by `super`. Finally we call `get_loss` method in `construct`. `reduction` has three legal parameters, `mean`, `sum` and `none`, which represent average, sum and original value.

## Loss Function and Model Training

Now we train model by the defined L1Loss.

### Defining Dataset and Network

Taking the simple linear function fitting as an example. The dataset and network structure is defined as follows:

> For a detailed introduction of linear fitting, please refer to the tutorial [Implementing Simple Linear Function Fitting](https://www.mindspore.cn/tutorials/en/r1.6/linear_regression.html).

1. Defining the Dataset

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

2. Defining the Network

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

### Training Model

`Model` is a MindSpore high level API which is for training, evaluating and inferring a model. After creating a dataset and defining `Model`, we can train the model by API `train`. Then we will train the model by `Model`, and use the defined  `L1Loss` as loss function.

1. Defining forward network, loss function and optimizer

    We will use the defined `LinearNet`  and `L1Loss` as forward network and loss function, and choose MindSpore's `Momemtum` as optimizer.

    ```python
    # define network
    net = LinearNet()
    # define loss function
    loss = L1Loss()
    # define optimizer
    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    ```

2. Defining `Model`

    When defining `Model`, it specifies the forward network, loss function and optimizer. The `Model` will associate them internally to form a training network.

    ```python
    from mindspore import Model

    # define Model
    model = Model(net, loss, opt)
    ```

3. Creating dataset, and calling `train` to train the model

    When calling the train interface, you must specify the number of iterations `epoch` and the training dataset `train_dataset`. We set `epoch` to 1, and use the dataset created by `create_dataset` as the training set. `callbacks` is an optional parameter of the `train` interface. `LossMonitor` can be used in `callbacks` to monitor the change of the loss function value during the training process. `dataset_sink_mode` is also an optional parameter, here is set to False, which means to use non-sink mode for training.

    ```python
    from mindspore.train.callback import LossMonitor

    # create dataset
    ds_train = create_dataset(num_data=160)
    # training
    model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
    ```

The complete code is as follows:

> In the following example, the parameter initialization uses random values, and the output results in specific execution may be different from the results of local execution; if you need to stabilize the output of a fixed value, you can set a fixed random seed. For the setting method, please refer to [mindspore.set_seed()](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.set_seed.html).

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

The output is as follows:

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

## Multilabel Loss Function and Model Training

In the last chapter, we defined a simple loss function `L1Loss`. Writing other loss functions is similar to `L1Loss`. However, some deep learning datasets are complex, such as the object detection network Faster R-CNN's dataset, which has several labels rather than simple data or label. The definition and usage of loss function is different in this situation.

Faster R-CNN's structure is too complex to detailed describe here. This chapter will expand the linear function fitting by creating a multilabel dataset. Then we will introduce how to define loss function and train by `Model`.

### Defining Multilabel Dataset

Firstly we define the dataset and make a slight modification to it:

1. `get_multilabel_data` will output two labels,`y1` and `y2`.
2. The parameters of `column_names` of `GeneratorDataset` are ['data', 'label1', 'label2']

Then `create_multilabel_dataset` will create dataset which has one `data`, and two labels `label1` and `label2`.

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

### Defining Multilabel Loss Function

We will define a loss function `L1LossForMultiLabel` according to defined multilabel dataset. The inputs of loss function's `construct` are predicted value `base`, and true value `target1` and `target2`. We will calculate the error between predict value and `target1`, `target2` respectively, and take the average of two values as final loss. The code is as follow:

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

### Training Multilabel Model

Model will internally link the forward network, loss function and optimizer. Forward network is connected to loss function by `nn.WithLossCell`, and forward network is connected to loss function by`nn.WithLossCell` as follows:

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

It should be noted that the default `nn.WithLossCell` of normal `Model` only has two inputs `data` and `label` , which is not suitable for multilabel case. Users need to connect the forward network and loss function as follows if they want to train by `Model`.

1. Defining the suitable `CustomWithLossCell` in this case

    We can copy the definition of `nn.WithLossCell` by changing the input of the `construct` to three parameters, that is, passing data to `backend`, and predicted and true value to`loss_fn`.

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

2. Connecting the forward network and loss function by `CustomWithLossCell`

    We use the forward network `LinearNet` defined in last chapter, and loss function `L1LossForMultiLabel`. Then connecting them by `CustomWithLossCell` as follows:

    ```python
    net = LinearNet()
    loss = L1LossForMultiLabel()
    loss_net = CustomWithLossCell(net, loss)
    ```

    `loss_net` contains the logic of forward network and loss function.

3. Defining Model and Training

    The `network` of `Model` is set to `loss_net`. `loss_fn` is not appointed, while the optimizer is still `Momentum`. As the user do not appoint `loss_fn`, `Model` will know that `network` has its own loss function logit. And it will not encapsulate forward network and loss function by `nn.WithLossCell`.

    Creating multilabel dataset by `create_multilabel_dataset` and training:

    ```python
    from mindspore.train.callback import LossMonitor
    from mindspore import Model

    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    model = Model(network=loss_net, optimizer=opt)
    ds_train = create_multilabel_dataset(num_data=160)
    model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
    ```

The complete code is as follows:

> In the following example, the parameter initialization uses random values, and the output results in specific execution may be different from the results of local execution; if you need to stabilize the output of a fixed value, you can set a fixed random seed. For the setting method, please refer to [mindspore.set_seed()](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.set_seed.html).

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

The output is as follow:

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

This chapter explains how to define loss function and train by `Model` in multilabel case. In some other cases, we can train the model by similar ways.
