# Model Training

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/train.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Model training is generally divided into four steps:

1. Build a dataset.
2. Define a neural network.
3. Define hyperparameters, a loss function, and an optimizer.
4. Input dataset for training and evaluation.

Through the above chapters, we have learned how to build datasets and build models. This chapter will focus on how to set up superparameters and train models. This chapter introduces training LeNet network using MNIST dataset taking MNIST dataset and LeNet network as an example.

Load the previous code from the chapters Data Processing and Building Networks.

```python
import os
import requests
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.nn as nn
from mindspore.common.initializer import Normal

requests.packages.urllib3.disable_warnings()


def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url),
                                                                                       path))


train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

# Downloading dataset
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte",
                 train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte",
                 train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte",
                 test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte",
                 test_path)


def create_dataset(data_path, batch_size=16, image_size=32):
    # Loading MINIST dataset
    dataset = ds.MnistDataset(data_path)

    rescale = 1.0 / 255.0
    shift = 0.0

    # Image transform operations
    trans = [
        vision.Resize(size=image_size),
        vision.Rescale(rescale, shift),
        vision.HWC2CHW(),
    ]
    dataset = dataset.map(operations=trans, input_columns=["image"])
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


# Customize networks
class LeNet(nn.Cell):
    def __init__(self, num_classes=10, num_channel=1):
        super(LeNet, self).__init__()
        layers = [nn.Conv2d(num_channel, 6, 5, pad_mode='valid'),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(6, 16, 5, pad_mode='valid'),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Flatten(),
                  nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02)),
                  nn.ReLU(),
                  nn.Dense(120, 84, weight_init=Normal(0.02)),
                  nn.ReLU(),
                  nn.Dense(84, num_classes, weight_init=Normal(0.02))]
        # Network management with CellList
        self.build_block = nn.CellList(layers)

    def construct(self, x):
        # for loop execution network
        for layer in self.build_block:
            x = layer(x)
        return x


# Define neural networks
network = LeNet()
```

## Hyper-parametric

Hyperparameters can be adjusted to control the model training and optimization process. Different hyperparameter values may affect the model training and convergence speed. Currently, deep learning models are optimized using the batch stochastic gradient descent algorithm. The principle of the stochastic gradient descent algorithm is as follows:
$w_{t+1}=w_{t}-\eta \frac{1}{n} \sum_{x \in \mathcal{B}} \nabla l\left(x, w_{t}\right)$

In the formula, $n$ is the batch size, and $η$ is a learning rate. In addition, $w_{t}$ is the weight parameter in the training batch t, and $\nabla l$ is the derivative of the loss function. In addition to the gradient itself, the two factors directly determine the weight update of the model. From the perspective of the optimization itself, the two factors are the most important parameters that affect the convergence of the model performance. Generally, the following hyperparameters are defined for training:

Epoch: specifies number of times that the dataset is traversed during training.

Batch size: specifies the size of each batch of data to be read. If the batch size is too small, it takes a long time and the gradient oscillation is serious, which is unfavorable to convergence. If the batch size is too large, the gradient directions of different batches do not change, and the local minimum value is easy to fall into. Therefore, you need to select a proper batch size to effectively improve the model precision and global convergence.

Learning rate: If the learning rate is low, the convergence speed slows down. If the learning rate is high, unpredictable results such as no training convergence may occur. Gradient descent is a parameter optimization algorithm that is widely used to minimize model errors. The gradient descent estimates the parameters of the model by iterating and minimizing the loss function at each step. The learning rate is used to control the learning progress of a model during iteration.

```python
epochs = 10
batch_size = 32
momentum = 0.9
learning_rate = 1e-2
```

## Defining Loss Functions

The **loss function** is used to evaluate the difference between **predicted value** and **target value** of a model. Here, use `SoftmaxCrossEntropyWithLogits` to calculate the cross entropy between the predicted and true values. The code example is as follows:

```python
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
```

## Defining Optimizer

An **optimizer function** is used to compute and update the gradient. The selection of the model optimization algorithm directly affects the performance of the final model. A poor effect may be caused by the optimization algorithm instead of the feature or model design. Here, the Momentum optimizer is used. `mindspore.nn` provides many common optimizer functions, such as `Adam`, `SGD` and `RMSProp`.

You need to build an `Optimizer` object. This object can update parameters based on the computed gradient. To build an `Optimizer`, you need to provide an optimizable parameter, for example, all trainable parametres in the network, i.e. set the entry for the optimizer to `network.trainable_params()`.

Then, you can set the `Optimizer` parameter options, such as the learning rate and weight decay. A code example is as follows:

```python
net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=0.9)
```

## Defining Training Process

Firstly, we define the forward network `forward`, which outputs the loss values of the forward propagation network; then we define the single-step training process `train_step`, which achieves the update of the network weights by back propagation. Finally, the training process `train_epoch` is defined, and the training dataset and model are passed in to realize the training of a single epoch.

```python
import mindspore as ms
import mindspore.ops as ops


# Forward network
def forward(inputs, targets):
    outputs = network(inputs)
    loss = net_loss(outputs, targets)
    return loss


grad_fn = ops.value_and_grad(forward, None, net_opt.parameters)


# Defining a single-step training process to update the network weights
def train_step(inputs, targets):
    loss, grads = grad_fn(inputs, targets)
    net_opt(grads)
    return loss


# Defining a single-epoch training process
def train_epoch(dataset, network):
    # Setting the network to training mode
    network.set_train()
    # Training the network by using dataset
    for data in dataset.create_dict_iterator():
        loss = train_step(ms.Tensor(data['image'], ms.float32), ms.Tensor(data["label"], ms.int32))

    return loss
```

## Defining Verification Process

Define the verification process `evaluate_epoch`, pass in the dataset to be verified and the trained network, and calculate the accuracy of the network classification.

```python
import numpy as np


def evaluate_epoch(dataset, network):
    size = dataset.get_dataset_size()
    accuracy = 0

    for data in dataset.create_dict_iterator():
        output = network(ms.Tensor(data['image'], ms.float32))
        acc = np.equal(output.argmax(1).asnumpy(), data["label"].asnumpy())
        accuracy += np.mean(acc)
    accuracy /= size

    return accuracy
```

## Model Training and Evaluation

After defining the training process and validation process, the model can be trained and evaluated after loading the training dataset and validation dataset.

```python
import mindspore as ms
import mindspore.ops as ops
import time

# Loading training dataset
dataset_train = create_dataset(train_path, batch_size)
# Loading test dataset
dataset_test = create_dataset(test_path, batch_size)

for epoch in range(epochs):
    start = time.time()
    result = train_epoch(dataset_train, network)
    acc = evaluate_epoch(dataset_test, network)
    end = time.time()
    time_ = round(end - start, 2)

    print("-" * 20)
    print(f"Epoch:[{epoch}/{epochs}], "
          f"Train loss:{result.asnumpy():5.4f}, "
          f"Accuracy:{acc:5.3f}, "
          f"Time:{time_}s, ")
```

```text
[EVENT] (PID29692) 2022-09-08-10:55:46.568.557 Start compiling 'after_grad' and it will take a while. Please wait...
[EVENT] (PID29692) 2022-09-08-10:55:46.695.443 End compiling 'after_grad'.
[EVENT] (PID29692) 2022-09-08-10:55:46.723.607 Start compiling 'Momentum.construct' and it will take a while. Please wait...
[EVENT] (PID29692) 2022-09-08-10:55:46.764.006 End compiling 'Momentum.construct'.
[EVENT] (PID29692) 2022-09-08-10:55:51.390.778 Start compiling 'LeNet.construct' and it will take a while. Please wait...
[EVENT] (PID29692) 2022-09-08-10:55:51.428.831 End compiling 'LeNet.construct'.
--------------------
Epoch:[0/10], Train loss:2.3203, Accuracy:0.113, Time:5.26s,
--------------------
Epoch:[1/10], Train loss:0.0699, Accuracy:0.956, Time:4.84s,
--------------------
Epoch:[2/10], Train loss:0.0157, Accuracy:0.976, Time:4.63s,
--------------------
Epoch:[3/10], Train loss:0.0570, Accuracy:0.974, Time:4.54s,
--------------------
Epoch:[4/10], Train loss:0.0255, Accuracy:0.986, Time:4.64s,
--------------------
Epoch:[5/10], Train loss:0.2859, Accuracy:0.986, Time:4.74s,
--------------------
Epoch:[6/10], Train loss:0.0005, Accuracy:0.985, Time:4.67s,
--------------------
Epoch:[7/10], Train loss:0.0166, Accuracy:0.989, Time:4.73s,
--------------------
Epoch:[8/10], Train loss:0.0046, Accuracy:0.988, Time:4.61s,
--------------------
Epoch:[9/10], Train loss:0.0003, Accuracy:0.987, Time:4.67s,
```

The loss values are printed during the training process, and the loss values fluctuate, but in general the loss values gradually decrease and the accuracy increases. The loss values of each individual are random and not necessarily the same.