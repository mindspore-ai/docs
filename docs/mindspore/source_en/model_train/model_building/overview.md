# Model Building Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_en/model_train/model_building/overview.md)

## Model Building

In the MindSpore framework, neural network models are constructed by combining neural network layers and Tensor operations, where the mindspore.nn module provides a rich set of common neural network layer implementations. The core of the framework is the Cell class, which is both the base class for building all networks and the basic unit of the network. A neural network model is abstracted as a Cell object, which is composed of multiple sub-Cells in an ordered fashion, forming a hierarchical nested structure. This design allows users to efficiently build and manage complex neural network architectures utilizing object-oriented programming ideas.

## Defining Model Class

User-defined neural networks typically inherit from the `mindspore.nn.Cell` class. In the inherited subclasses, the `__init__` method is used to instantiate sub-Cells (e.g., convolutional layers, pooling layers, etc.) and to perform related state management, such as parameter initialization. And the `construct` method defines the specific calculation logic. Please refer to [Functional and Cell](https://www.mindspore.cn/docs/en/r2.5.0/model_train/model_building/functional_and_cell.html) for detailed usage.

MindSpore builds the LeNet5 model as shown below:

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        '''
        Forward network.
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Neural Network Layer

MindSpore encapsulates a variety of common neural network layers. Users can look up the desired neural network layer in [mindspore.nn](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore.nn.html). For example, in the field of image processing, the `nn.Conv2d` layer provides convenient support for convolutional operations; and `nn.ReLU`, as a nonlinear activation layer, can effectively increase the network nonlinear expressive capability. These predefined neural network layers greatly simplify the complexity of network construction, allowing users to focus more on model design and optimization.

## Model Parameter

The core of a neural network model lies in its internal neural network layers, which not only define the forward propagation path of the data, but also contain trainable weight parameters and bias parameters, such as nn.Dense. These parameters are the cornerstones of model learning and are continuously optimized during the training process by backpropagation algorithms to minimize the loss function and improve the model performance.

MindSpore provides a convenient interface to manage these parameters. Users can get the parameter names of the model and their corresponding values by calling `parameters_dict`, `get_parameters`, and `trainable_params`. of the model instance. Please refer to [Tensor and Parameter](https://www.mindspore.cn/docs/en/r2.5.0/model_train/model_building/tensor_and_parameter.html) for details.