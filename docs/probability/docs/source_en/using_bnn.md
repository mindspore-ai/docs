# Using BNN to Implement an Image Classification Application

<a href="https://gitee.com/mindspore/docs/blob/master/docs/probability/docs/source_en/using_bnn.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Deep learning models have a strong fitting capability, while Bayesian theory has a good explainable capability. MindSpore Deep Probability Programming combines deep learning and Bayesian learning. By setting the network weight to distribution, introducing hidden space distribution, etc., the distribution can be sampled forward propagation, which introduces uncertainty and enhances The robustness and interpretability of the model are improved.

This chapter will introduce in detail the application of Bayesian neural network in deep probabilistic programming on MindSpore. Before starting to practice, make sure that you have correctly installed MindSpore 0.7.0-beta and above.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/master/tests/st/probability/bnn_layers>.  
> BNN only supports GRAPH mode now, please set `context.set_context(mode=context.GRAPH_MODE)` in your code.

## Using BNN

BNN is a basic model composed of probabilistic model and neural network. Its weight is not a definite value, but a distribution. The following example describes how to use the `bnn_layers` module in MDP to implement a BNN, and then use the BNN to implement a simple image classification function. The overall process is as follows:

1. Process the MNIST dataset.
2. Define the Bayes LeNet.
3. Define the loss function and optimizer.
4. Load and train the dataset.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/master/tests/st/probability/bnn_layers>.
> BNN only supports GRAPH mode now, please set `context.set_context(mode=context.GRAPH_MODE)` in your code.

## Environment Preparation

Set the training mode to graph mode and the computing platform to GPU.

```python
from mindspore import set_context, GRAPH_MODE

set_context(mode=GRAPH_MODE, save_graphs=False, device_target="GPU")
```

## Data Preparation

### Downloading the Dataset

Download the MNIST dataset and unzip it to the specified location, execute the following command:

```python
import os
import requests

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

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)
```

```text
./datasets/MNIST_Data
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte

2 directories, 4 files
```

### Defining the Dataset Enhancement Method

The original training dataset of the MNIST dataset is 60,000 single-channel digital images with $28\times28$ pixels. The LeNet5 network containing the Bayesian layer used in this training received the training data tensor as `(32,1 ,32,32)`, through the custom create_dataset function to enhance the original dataset to meet the training requirements of the data, the specific enhancement operation explanation can refer to [Quick Start for Beginners](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html).

```python
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dataset as ds

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define some parameters needed for data enhancement and rough justification
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # according to the parameters, generate the corresponding data enhancement method
    c_trans = [
        CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR),
        CV.Rescale(rescale_nml, shift_nml),
        CV.Rescale(rescale, shift),
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(mstype.int32)

    # using map to apply operations to a dataset
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=num_parallel_workers)

    # process the generated dataset
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds
```

### Defining the BNN

In the classic LeNet5 network, the data goes through the following calculation process: convolution 1->activation->pooling->convolution 2->activation->pooling->dimensionality reduction->full connection 1->full connection 2-> Fully connected 3.

In this example, a probabilistic programming method will be introduced, using the `bnn_layers` module to transform the convolutional layer and the fully connected layer into a Bayesian layer.

```python
import mindspore.nn as nn
from mindspore.nn.probability import bnn_layers
import mindspore.ops as ops
from mindspore import dtype as mstype


class BNNLeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(BNNLeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = bnn_layers.ConvReparam(1, 6, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.conv2 = bnn_layers.ConvReparam(6, 16, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.fc1 = bnn_layers.DenseReparam(16 * 5 * 5, 120)
        self.fc2 = bnn_layers.DenseReparam(120, 84)
        self.fc3 = bnn_layers.DenseReparam(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = BNNLeNet5(num_class=10)
for layer in network.trainable_params():
    print(layer.name)
```

```text
conv1.weight_posterior.mean
conv1.weight_posterior.untransformed_std
conv2.weight_posterior.mean
conv2.weight_posterior.untransformed_std
fc1.weight_posterior.mean
fc1.weight_posterior.untransformed_std
fc1.bias_posterior.mean
fc1.bias_posterior.untransformed_std
fc2.weight_posterior.mean
fc2.weight_posterior.untransformed_std
fc2.bias_posterior.mean
fc2.bias_posterior.untransformed_std
fc3.weight_posterior.mean
fc3.weight_posterior.untransformed_std
fc3.bias_posterior.mean
fc3.bias_posterior.untransformed_std
```

The printed information shows that the convolutional layer and the fully connected layer of the LeNet network constructed with the `bnn_layers` module are both Bayesian layers.

### Defining the Loss Function and Optimizer

Next, you need to define the loss function and the optimizer. The loss function is the training target of deep learning, also called the objective function. It can be understood as the distance between the output of the neural network (Logits) and the label (Labels), which is a scalar data.

Common loss functions include mean square error, L2 loss, Hinge loss, and cross entropy. Cross entropy is usually used for image classification.

The optimizer is used for neural network solution (training). Because of the large scale of neural network parameters, the stochastic gradient descent (SGD) algorithm and its improved algorithm are used in deep learning to solve the problem. MindSpore encapsulates common optimizers, such as `SGD`, `Adam`, and `Momemtum`. In this example, the `Adam` optimizer is used. Generally, two parameters need to be set: learning rate (`learning_rate`) and weight attenuation (`weight_decay`).

An example of the code for defining the loss function and optimizer in MindSpore is as follows:

```python
import mindspore.nn as nn

# loss function definition
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = nn.AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
```

### Training the Network

The training process of BNN is similar to that of DNN. The only difference is that `WithLossCell` is replaced with `WithBNNLossCell` applicable to BNN. In addition to the `backbone` and `loss_fn` parameters, the `dnn_factor` and `bnn_factor` parameters are added to `WithBNNLossCell`. `dnn_factor` is a coefficient of the overall network loss computed by a loss function, and `bnn_factor` is a coefficient of the KL divergence of each Bayesian layer. The two parameters are used to balance the overall network loss and the KL divergence of the Bayesian layer, preventing the overall network loss from being covered by a large KL divergence.

- `dnn_factor` is the coefficient of the overall network loss calculated by the loss function.
- `bnn_factor` is the coefficient of the KL divergence of each Bayesian layer.

The code examples of `train_model` and `validate_model` in MindSpore are as follows:

```python
def train_model(train_net, net, dataset):
    accs = []
    loss_sum = 0
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].asnumpy().astype(np.float32))
        label = Tensor(data['label'].asnumpy().astype(np.int32))
        loss = train_net(train_x, label)
        output = net(train_x)
        log_output = ops.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)
        loss_sum += loss.asnumpy()

    loss_sum = loss_sum / len(accs)
    acc_mean = np.mean(accs)
    return loss_sum, acc_mean


def validate_model(net, dataset):
    accs = []
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].asnumpy().astype(np.float32))
        label = Tensor(data['label'].asnumpy().astype(np.int32))
        output = net(train_x)
        log_output = ops.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)

    acc_mean = np.mean(accs)
    return acc_mean
```

Perform training.

```python
from mindspore.nn import TrainOneStepCell
from mindspore import Tensor
import numpy as np

net_with_loss = bnn_layers.WithBNNLossCell(network, criterion, dnn_factor=60000, bnn_factor=0.000001)
train_bnn_network = TrainOneStepCell(net_with_loss, optimizer)
train_bnn_network.set_train()

train_set = create_dataset('./datasets/MNIST_Data/train', 64, 1)
test_set = create_dataset('./datasets/MNIST_Data/test', 64, 1)

epoch = 10

for i in range(epoch):
    train_loss, train_acc = train_model(train_bnn_network, network, train_set)

    valid_acc = validate_model(network, test_set)

    print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tvalidation Accuracy: {:.4f}'.
          format(i+1, train_loss, train_acc, valid_acc))
```

```text
Epoch: 1    Training Loss: 21444.8605   Training Accuracy: 0.8928    validation Accuracy: 0.9513
Epoch: 2    Training Loss: 9396.3887    Training Accuracy: 0.9536    validation Accuracy: 0.9635
Epoch: 3    Training Loss: 7320.2412    Training Accuracy: 0.9641    validation Accuracy: 0.9674
Epoch: 4    Training Loss: 6221.6970    Training Accuracy: 0.9685    validation Accuracy: 0.9731
Epoch: 5    Training Loss: 5450.9543    Training Accuracy: 0.9725    validation Accuracy: 0.9733
Epoch: 6    Training Loss: 4898.9741    Training Accuracy: 0.9754    validation Accuracy: 0.9767
Epoch: 7    Training Loss: 4505.7502    Training Accuracy: 0.9775    validation Accuracy: 0.9784
Epoch: 8    Training Loss: 4099.8783    Training Accuracy: 0.9797    validation Accuracy: 0.9791
Epoch: 9    Training Loss: 3795.2288    Training Accuracy: 0.9810    validation Accuracy: 0.9796
Epoch: 10    Training Loss: 3581.4254    Training Accuracy: 0.9823    validation Accuracy: 0.9773
```
