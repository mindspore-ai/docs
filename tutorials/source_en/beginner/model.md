# Building a Network

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

A neural network model consists of multiple data operation layers. `mindspore.nn` provides various basic network modules. The following uses LeNet-5 as an example to implement handwritten digit recognition task in deep learning.

## LeNet-5 Model

[LeNet-5](https://ieeexplore.ieee.org/document/726791) is a typical convolutional neural network proposed by professor Yann LeCun in 1998, which achieves 99.4% accuracy on the MNIST dataset and is the first classic in the field of CNN. The model structure is shown in the following figure:

![LeNet-5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/lenet.png)

In the preceding figure, C indicates the convolutional layer layer, S indicates the sampling layer, and F indicates the fully-connected layer.

According to the network structure of LeNet, except the input layer, LeNet contains seven layers: three convolutional layers, two subsampling layers, and two fully-connected layers.

The input size of an image is fixed at $32 \times 32$. To achieve a good convolution effect, the number must be in the center of the image. Therefore, the input $32 \times 32$ is the result after the image is filled with $28 \times 28$. Unlike the three-channel input images of the CNN network, the input images of LeNet are only normalized binary images. The output of the network is the prediction probability of digits 0 to 9, which can be understood as the probability that the input image belongs to digits 0 to 9.

## Defining a Model Class

The `Cell` class of MindSpore is the base class for building all networks and the basic unit of a network. When a neural network is required, you need to inherit the `Cell` class and overwrite the `__init__` and `construct` methods.

The `mindspore.nn` class is the base class for building all networks and the basic unit of a network. When need to customize the network, you can inherit the `nn.Cell` class and overwrite the `__init__` and `construct` methods.

To facilitate the management and composition of more complex networks, `mindspore.nn` provides containers to manage the submodel blocks or model layers in the network, `nn.CellList` and `nn.SequentialCell`. Here, we have chosen the `nn.CellList` method.

```python
from mindspore import nn
from mindspore.common.initializer import Normal

# Customize the network
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
        # Execute the network with a for loop
        for layer in self.build_block:
            x = layer(x)
        return x
```

Next, build the neural network model defined above and look at the structure of the network model.

```python
model = LeNet()

print(model)
```

```text
LeNet<
  (build_block): CellList<
    (0): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
    (1): ReLU<>
    (2): MaxPool2d<kernel_size=2, stride=2, pad_mode=VALID>
    (3): Conv2d<input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
    (4): ReLU<>
    (5): MaxPool2d<kernel_size=2, stride=2, pad_mode=VALID>
    (6): Flatten<>
    (7): Dense<input_channels=400, output_channels=120, has_bias=True>
    (8): ReLU<>
    (9): Dense<input_channels=120, output_channels=84, has_bias=True>
    (10): ReLU<>
    (11): Dense<input_channels=84, output_channels=10, has_bias=True>
    >
  >
```

## Model Layers

The following describes the key member functions of the `Cell` class used in LeNet-5, and then describes how to use the `Cell` class to access model parameters through the instantiation network. For more information about the `Cell` class, see [mindspore.nn interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html).

### nn.Conv2d

Add the `nn.Conv2d` layer and add a convolution function to the network to help the neural network extract features.

```python
import numpy as np

import mindspore as ms

# The number of channels input is 1, the number of channels of output is 6, the convolutional kernel size is 5 x 5, and the parameters are initialized by using the normal operator, and the pixels are not filled.
conv2d = nn.Conv2d(1, 6, 5, has_bias=False, weight_init='normal', pad_mode='same')
input_x = ms.Tensor(np.ones([1, 1, 32, 32]), ms.float32)

print(conv2d(input_x).shape)
```

```text
(1, 6, 32, 32)
```

### nn.ReLU

Add the `nn.ReLU` layer and add a non-linear activation function to the network to help the neural network learn various complex features.

```python
relu = nn.ReLU()

input_x = ms.Tensor(np.array([-1, 2, -3, 2, -1]), ms.float16)

output = relu(input_x)
print(output)
```

```text
[0. 2. 0. 2. 0.]
```

### nn.MaxPool2d

Initialize the `nn.MaxPool2d` layer and down-sample the 6 x 28 x 28 tensor to a 6 x 7 x 7 tensor.

```text
max_pool2d = nn.MaxPool2d(kernel_size=4, stride=4)
input_x = ms.Tensor(np.ones([1, 6, 28, 28]), ms.float32)

print(max_pool2d(input_x).shape)
```

### nn.Flatten

Initialize the `nn.Flatten` layer and convert the 1 x 16 x 5 x 5 4D tensor into 400 consecutive 2D tensors.

```python
flatten = nn.Flatten()
input_x = ms.Tensor(np.ones([1, 16, 5, 5]), ms.float32)
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
input_x = ms.Tensor(np.ones([1, 400]), ms.float32)
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
layer:build_block.0.weight, shape:(6, 1, 5, 5), dtype:Float32, requeires_grad:True
layer:build_block.3.weight, shape:(16, 6, 5, 5), dtype:Float32, requeires_grad:True
layer:build_block.7.weight, shape:(120, 400), dtype:Float32, requeires_grad:True
layer:build_block.7.bias, shape:(120,), dtype:Float32, requeires_grad:True
layer:build_block.9.weight, shape:(84, 120), dtype:Float32, requeires_grad:True
layer:build_block.9.bias, shape:(84,), dtype:Float32, requeires_grad:True
layer:build_block.11.weight, shape:(10, 84), dtype:Float32, requeires_grad:True
layer:build_block.11.bias, shape:(10,), dtype:Float32, requeires_grad:True
```
