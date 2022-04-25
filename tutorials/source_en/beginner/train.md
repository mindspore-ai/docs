# Model Training

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/source_en/beginner/train.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

After learning how to create a model and build a dataset in the preceding tutorials, you can start to learn how to set hyperparameters and optimize model parameters.

## Hyperparameters

Hyperparameters can be adjusted to control the model training and optimization process. Different hyperparameter values may affect the model training and convergence speed. Currently, deep learning models are optimized using the batch stochastic gradient descent algorithm. The principle of the stochastic gradient descent algorithm is as follows: $w_{t+1}=w_{t}-\eta \frac{1}{n} \sum_{x \in \mathcal{B}} \nabla l\left(x, w_{t}\right)$

In the formula, $n$ is the batch size, and $η$ is a learning rate. In addition, $w_{t}$ is the weight parameter in the training batch t, and $\nabla l$ is the derivative of the loss function. In addition to the gradient itself, the two factors directly determine the weight update of the model. From the perspective of the optimization itself, the two factors are the most important parameters that affect the convergence of the model performance. Generally, the following hyperparameters are defined for training:

Epoch: specifies number of times that the dataset is traversed during training.

Batch size: specifies the size of each batch of data to be read. If the batch size is too small, it takes a long time and the gradient oscillation is serious, which is unfavorable to convergence. If the batch size is too large, the gradient directions of different batches do not change, and the local minimum value is easy to fall into. Therefore, you need to select a proper batch size to effectively improve the model precision and global convergence.

Learning rate: If the learning rate is low, the convergence speed slows down. If the learning rate is high, unpredictable results such as no training convergence may occur. Gradient descent is a parameter optimization algorithm that is widely used to minimize model errors. The gradient descent estimates the parameters of the model by iterating and minimizing the loss function at each step. The learning rate is used to control the learning progress of a model during iteration.

![learning-rate](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/learning_rate.png)

```python
epochs = 10
batch_size = 32
momentum = 0.9
learning_rate = 1e-2
```

## Loss Functions

The **loss function** is used to evaluate the difference between **predicted value** and **target value** of a model. Here, the absolute error loss function `L1Loss` is used: $\text { L1 Loss Function }=\sum_{i=1}^{n}\left|y_{true}-y_{predicted}\right|$

`mindspore.nn.loss` provides many common loss functions, such as `SoftmaxCrossEntropyWithLogits`, `MSELoss`, and `SmoothL1Loss`.

Given the predicted value and the target value, we calculate the error (loss value) between the predicted value and the target value by a loss function. The method is as follows:

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

## Optimizer Functions

An optimizer is used to compute and update the gradient. The selection of the model optimization algorithm directly affects the performance of the final model. A poor effect may be caused by the optimization algorithm instead of the feature or model design.

All optimization logic of MindSpore is encapsulated in the `Optimizer` object. Here, the Momentum optimizer is used. `mindspore.nn` provides many common optimizers, such as `Adam`, `SGD` and `RMSProp`.

You need to build an `Optimizer` object. This object can retain the current parameter status and update parameters based on the computed gradient. To build an `Optimizer`, you need to provide an iterator that contains parameters (must be variable objects) to be optimized. For example, set parameters to `net.trainable_params()` for all `parameter` that can be trained on the network.

Then, you can set the `Optimizer` parameter options, such as the learning rate and weight decay. A code example is as follows:

```python
from mindspore import nn
from mindvision.classification.models import lenet

net = lenet(num_classes=10, pretrained=False)
optim = nn.Momentum(net.trainable_params(), learning_rate, momentum)
```

## Model Training

Model training consists of four steps:

1. Build a dataset.
2. Define a neural network.
3. Define hyperparameters, a loss function, and an optimizer.
4. Enter the epoch and dataset for training.

The model training sample code is as follows:

```python
import mindspore.nn as nn
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor

# 1. Build a dataset.
download_train = Mnist(path="./mnist", split="train", batch_size=batch_size, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

# 2. Define a neural network.
network = lenet(num_classes=10, pretrained=False)
# 3.1 Define a loss function.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 3.2 Define an optimizer function.
net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=momentum)
# 3.3 Initialize model parameters.
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})

# 4. Train the neural network.
model.train(epochs, dataset_train, callbacks=[LossMonitor(learning_rate, 1875)])
```

```text
Epoch:[  0/ 10], step:[ 1875/ 1875], loss:[0.189/1.176], time:2.254 ms, lr:0.01000
Epoch time: 4286.163 ms, per step time: 2.286 ms, avg loss: 1.176
Epoch:[  1/ 10], step:[ 1875/ 1875], loss:[0.085/0.080], time:1.895 ms, lr:0.01000
Epoch time: 4064.532 ms, per step time: 2.168 ms, avg loss: 0.080
Epoch:[  2/ 10], step:[ 1875/ 1875], loss:[0.021/0.054], time:1.901 ms, lr:0.01000
Epoch time: 4194.333 ms, per step time: 2.237 ms, avg loss: 0.054
Epoch:[  3/ 10], step:[ 1875/ 1875], loss:[0.284/0.041], time:2.130 ms, lr:0.01000
Epoch time: 4252.222 ms, per step time: 2.268 ms, avg loss: 0.041
Epoch:[  4/ 10], step:[ 1875/ 1875], loss:[0.003/0.032], time:2.176 ms, lr:0.01000
Epoch time: 4216.039 ms, per step time: 2.249 ms, avg loss: 0.032
Epoch:[  5/ 10], step:[ 1875/ 1875], loss:[0.003/0.027], time:2.205 ms, lr:0.01000
Epoch time: 4400.771 ms, per step time: 2.347 ms, avg loss: 0.027
Epoch:[  6/ 10], step:[ 1875/ 1875], loss:[0.000/0.024], time:1.973 ms, lr:0.01000
Epoch time: 4554.252 ms, per step time: 2.429 ms, avg loss: 0.024
Epoch:[  7/ 10], step:[ 1875/ 1875], loss:[0.008/0.022], time:2.048 ms, lr:0.01000
Epoch time: 4361.135 ms, per step time: 2.326 ms, avg loss: 0.022
Epoch:[  8/ 10], step:[ 1875/ 1875], loss:[0.000/0.018], time:2.130 ms, lr:0.01000
Epoch time: 4547.597 ms, per step time: 2.425 ms, avg loss: 0.018
Epoch:[  9/ 10], step:[ 1875/ 1875], loss:[0.008/0.017], time:2.135 ms, lr:0.01000
Epoch time: 4601.861 ms, per step time: 2.454 ms, avg loss: 0.017
```

During the training, the loss value is printed and fluctuates. However, the loss value gradually decreases and the precision gradually increases. Loss values displayed each time may be different because of their randomness.
