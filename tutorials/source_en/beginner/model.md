# Building a Neural Network

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

A neural network model consists of multiple data operation layers. `mindspore.nn` provides various basic network modules. The following uses LeNet-5 as an example to first describe how to build a neural network model by using `mindspore.nn`, and then describes how to build a LeNet-5 model by using `mindvision.classification.models`.

> `mindvision.classification.models` is a network model interface developed based on `mindspore.nn`, providing some classic and commonly used network models for the convenience of users.

## LeNet-5 Model

[LeNet-5](https://ieeexplore.ieee.org/document/726791) is a typical convolutional neural network proposed by professor Yann LeCun in 1998, which achieves 99.4% accuracy on the MNIST dataset and is the first classic in the field of CNN. The model structure is shown in the following figure:

![LeNet-5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/lenet.png)

Except the input layer, LeNet contains seven layers: three convolutional layers, two subsampling layers, and two fully-connected layers.

## Defining a Model Class

In the preceding figure, C indicates the convolutional layer layer, S indicates the sampling layer, and F indicates the fully-connected layer.

The input size of an image is fixed at $32 x 32$. To achieve a good convolution effect, the number must be in the center of the image. Therefore, the input $32 x 32$ is the result after the image is filled with $28 x 28$. Unlike the three-channel input images of the CNN network, the input images of LeNet are only normalized binary images. The output of the network is the prediction probability of digits 0 to 9, which can be understood as the probability that the input image belongs to digits 0 to 9.

The `Cell` class of MindSpore is the base class for building all networks and the basic unit of a network. When a neural network is required, you need to inherit the `Cell` class and overwrite the `__init__` and `construct` methods.

```python
import mindspore.nn as nn

class LeNet5(nn.Cell):
    """
    LeNet-5 network structure
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # Convolutional layer, where the number of input channels is num_channel, the number of output channels is 6, and the convolutional kernel size is 5 x 5.
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # Convolutional layer, where the number of input channels is 6, the number of output channels is 16, and the convolutional kernel size is 5 x 5.
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        # Fully-connected layer, where the number of inputs is 16 x 5 x 5 and the number of outputs is 120.
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        # Fully-connected layer, where the number of inputs is 120 and the number of outputs is 84.
        self.fc2 = nn.Dense(120, 84)
        # Fully-connected layer, where the number of inputs is 84 and the number of classes is num_class.
        self.fc3 = nn.Dense(84, num_class)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Pooling layer
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Multidimensional arrays are flattened into one-dimensional arrays.
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Use the defined operation to build a forward network.
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
```

Next, build the neural network model defined above and look at the structure of the network model.

```python
model = LeNet5()

print(model)
```

```text
LeNet5<
(conv1): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
(conv2): Conv2d<input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
(fc1): Dense<input_channels=400, output_channels=120, has_bias=True>
(fc2): Dense<input_channels=120, output_channels=84, has_bias=True>
(fc3): Dense<input_channels=84, output_channels=10, has_bias=True>
(relu): ReLU<>
(max_pool2d): MaxPool2d<kernel_size=2, stride=2, pad_mode=VALID>
(flatten): Flatten<>
>
```

## Model Layers

The following describes the key member functions of the `Cell` class used in LeNet-5, and then describes how to use the `Cell` class to access model parameters through the instantiation network. For more information about the `Cell` class, see [mindspore.nn interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html).

### nn.Conv2d

Add the `nn.Conv2d` layer and add a convolution function to the network to help the neural network extract features.

```python
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype

# The number of channels input is 1, the number of channels of output is 6, the convolutional kernel size is 5 x 5, and the parameters are initialized using the normal operator, and the pixels are not filled.
conv2d = nn.Conv2d(1, 6, 5, has_bias=False, weight_init='normal', pad_mode='same')
input_x = Tensor(np.ones([1, 1, 32, 32]), mstype.float32)

print(conv2d(input_x).shape)
```

```text
(1, 6, 32, 32)
```

### nn.ReLU

Add the `nn.ReLU` layer and add a non-linear activation function to the network to help the neural network learn various complex features.

```python
relu = nn.ReLU()

input_x = Tensor(np.array([-1, 2, -3, 2, -1]), mstype.float16)

output = relu(input_x)
print(output)
```

```text
[0. 2. 0. 2. 0.]
```

### nn.MaxPool2d

Initialize the `nn.MaxPool2d` layer and down-sample the 6 x 28 x 28 array to a 6 x 14 x 14 array.

```python
max_pool2d = nn.MaxPool2d(kernel_size=4, stride=4)
input_x = Tensor(np.ones([1, 6, 28, 28]), mstype.float32)

print(max_pool2d(input_x).shape)
```

```text
(1, 6, 7, 7)
```

### nn.Flatten

Initialize the `nn.Flatten` layer and convert the 1 x 16 x 5 x 5 array into 400 consecutive arrays.

```python
flatten = nn.Flatten()
input_x = Tensor(np.ones([1, 16, 5, 5]), mstype.float32)
output = flatten(input_x)

print(output.shape)
```

```text
(1, 400)
```

### nn.Dense

Initialize the `nn.Dense` layer and perform linear transformation on the input matrix.

```python
dense = nn.Dense(400, 120, weight_init='normal')
input_x = Tensor(np.ones([1, 400]), mstype.float32)
output = dense(input_x)

print(output.shape)
```

```text
(1, 120)
```

## Model Parameters

After instantiation is performed on the convolutional layer and the fully-connected layer in the network, there are a weight parameter and an offset parameter. These parameters are continuously optimized in a training process. During training, you can use `get_parameters()` to view the name, shape, and data type of each network layer, and whether backward calculation is performed.

```python
for m in model.get_parameters():
    print(f"layer:{m.name}, shape:{m.shape}, dtype:{m.dtype}, requeires_grad:{m.requires_grad}")
```

```text
layer:backbone.conv1.weight, shape:(6, 1, 5, 5), dtype:Float32, requeires_grad:True
layer:backbone.conv2.weight, shape:(16, 6, 5, 5), dtype:Float32, requeires_grad:True
layer:backbone.fc1.weight, shape:(120, 400), dtype:Float32, requeires_grad:True
layer:backbone.fc1.bias, shape:(120,), dtype:Float32, requeires_grad:True
layer:backbone.fc2.weight, shape:(84, 120), dtype:Float32, requeires_grad:True
layer:backbone.fc2.bias, shape:(84,), dtype:Float32, requeires_grad:True
layer:backbone.fc3.weight, shape:(10, 84), dtype:Float32, requeires_grad:True
layer:backbone.fc3.bias, shape:(10,), dtype:Float32, requeires_grad:True
```

## Quickly Building a LeNet-5 Model

The preceding describes how to use `mindspore.nn.cell` to build a LeNet-5 model. The built network model API is available in `mindvision.classification.models`. You can also use the `lenet` API to directly build a LeNet-5 model.

```python
from mindvision.classification.models import lenet

# `num_classes` indicates the number of classes, and `pretrained` determines whether to train with the trained model.
model = lenet(num_classes=10, pretrained=False)

for m in model.get_parameters():
    print(f"layer:{m.name}, shape:{m.shape}, dtype:{m.dtype}, requeires_grad:{m.requires_grad}")
```

```text
layer:backbone.conv1.weight, shape:(6, 1, 5, 5), dtype:Float32, requeires_grad:True
layer:backbone.conv2.weight, shape:(16, 6, 5, 5), dtype:Float32, requeires_grad:True
layer:backbone.fc1.weight, shape:(120, 400), dtype:Float32, requeires_grad:True
layer:backbone.fc1.bias, shape:(120,), dtype:Float32, requeires_grad:True
layer:backbone.fc2.weight, shape:(84, 120), dtype:Float32, requeires_grad:True
layer:backbone.fc2.bias, shape:(84,), dtype:Float32, requeires_grad:True
layer:backbone.fc3.weight, shape:(10, 84), dtype:Float32, requeires_grad:True
layer:backbone.fc3.bias, shape:(10,), dtype:Float32, requeires_grad:True
```