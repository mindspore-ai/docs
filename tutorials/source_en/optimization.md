# Training the Model

`Ascend` `GPU` `CPU` `Beginner` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_en/optimization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

After learning how to create a model and build a dataset in the preceding tutorials, you can start to learn how to set hyperparameters and optimize model parameters.

## Hyperparameters

Hyperparameters can be adjusted to control the model training and optimization process. Different hyperparameter values may affect the model training and convergence speed.

Generally, the following hyperparameters are defined for training:

- Epoch: specifies number of times that the dataset is traversed during training.
- Batch size: specifies the size of each batch of data to be read.
- Learning rate: If the learning rate is low, the convergence speed slows down. If the learning rate is high, unpredictable results such as no training convergence may occur.

```python
epochs = 5
batch_size = 64
learning_rate = 1e-3
```

## Loss Functions

The **loss function** is used to evaluate the difference between **predicted value** and **actual value** of a model. Here, the absolute error loss function `L1Loss` is used. `mindspore.nn.loss` provides many common loss functions, such as `SoftmaxCrossEntropyWithLogits`, `MSELoss`, and `SmoothL1Loss`.

The output value and target value are provided to compute the loss value. The method is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

loss = nn.L1Loss()
output_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
print(loss(output_data, target_data))
```

```text
    1.5
```

## Optimizer

An optimizer is used to compute and update the gradient. The selection of the model optimization algorithm directly affects the performance of the final model. A poor effect may be caused by the optimization algorithm instead of the feature or model design. All optimization logic of MindSpore is encapsulated in the `Optimizer` object. Here, the Momentum optimizer is used. `mindspore.nn` provides many common optimizers, such as `Adam` and `Momentum`.

You need to build an `Optimizer` object. This object can retain the current parameter status and update parameters based on the computed gradient.

To build an `Optimizer`, we need to provide an iterator that contains parameters (must be variable objects) to be optimized. For example, set `params` to `net.trainable_params()` for all `parameter` that can be trained on the network. Then, you can set the `Optimizer` parameter options, such as the learning rate and weight attenuation.

A code example is as follows:

```python
from mindspore import nn

optim = nn.Momentum(net.trainable_params(), 0.1, 0.9)
```

## Training

A model training process is generally divided into four steps.

1. Define a neural network.
2. Build a dataset.
3. Define hyperparameters, a loss function, and an optimizer.
4. Enter the epoch and dataset for training.

Execute the following command to download and decompress the dataset to the specified location.

```bash
mkdir ./datasets
wget -N https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz --no-check-certificate
tar -zxvf cifar-10-binary.tar.gz -C ./datasets
```

The code example for model training is as follows:

```python
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import nn, Tensor, Model
from mindspore import dtype as mstype
from mindspore.train.callback import LossMonitor

DATA_DIR = "./datasets/cifar-10-batches-bin"

# Define a neural network.
class Net(nn.Cell):
    def __init__(self, num_class=10, num_channel=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

net = Net()
epochs = 5
batch_size = 64
learning_rate = 1e-3

# Build a dataset.
sampler = ds.SequentialSampler(num_samples=128)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# Convert the data type.
type_cast_op_image = C.TypeCast(mstype.float32)
type_cast_op_label = C.TypeCast(mstype.int32)
HWC2CHW = CV.HWC2CHW()
dataset = dataset.map(operations=[type_cast_op_image, HWC2CHW], input_columns="image")
dataset = dataset.map(operations=type_cast_op_label, input_columns="label")
dataset = dataset.batch(batch_size)

# Define hyperparameters, a loss function, and an optimizer.
optim = nn.Momentum(net.trainable_params(), learning_rate, 0.9)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
cb = LossMonitor()

# Enter the epoch and dataset for training.
model = Model(net, loss_fn=loss, optimizer=optim)
model.train(epoch=epochs, train_dataset=dataset, callbacks=cb)
```

The output is as follows:

```text
epoch: 1 step: 1, loss is 2.3025818
epoch: 1 step: 2, loss is 2.3025775
epoch: 2 step: 1, loss is 2.3025408
epoch: 2 step: 2, loss is 2.3025331
epoch: 3 step: 1, loss is 2.3024616
epoch: 3 step: 2, loss is 2.302457
epoch: 4 step: 1, loss is 2.3023522
epoch: 4 step: 2, loss is 2.3023558
epoch: 5 step: 1, loss is 2.3022182
epoch: 5 step: 2, loss is 2.3022337
```