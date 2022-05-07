# Implementing the Concept Drift Detection Application of Image Data

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/concept_drift_images.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Overview

Concept drift of image data is an important data phenomenon in the AI learning field. It is also called out-of-distribution (OOD), which indicates that the image data (real-time distribution) in online inference is inconsistent with the training data (historical distribution).
 For example, if the neural network model is obtained through training based on the MNIST dataset, but the actual test data is in the CIFAR-10 data environment, the CIFAR-10 dataset is an OOD sample.

This example provides a method for detecting a distribution change of image data. An overall process is as follows:

1. Load public datasets or use user-defined data.
2. Load a neural network model.
3. Initialize the image concept drift parameters.
4. Obtain an optimal concept drift detection threshold.
5. Execute the concept drift detection function.
6. View the execution result.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/mindarmour/blob/master/examples/reliability/concept_drift_check_images_lenet.py>.

## Preparations

Ensure that the MindSpore is correctly installed. If not, install MindSpore by following the [Installation Guide](https://www.mindspore.cn/install/en).

### Preparing a Dataset

The public image datasets MNIST and CIFAR-10 are used in the example.
> Dataset download pages: <http://yann.lecun.com/exdb/mnist/> and <http://www.cs.toronto.edu/~kriz/cifar.html>.

### Importing the Python Library and Modules

Before start, you need to import the Python library.

```python
import numpy as np
from mindspore import Tensor
from mindspore import Model
from mindarmour.utils import LogUtil
from mindspore import Model, nn
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5
from mindspore import load_checkpoint, load_param_into_net
from mindarmour.reliability import OodDetectorFeatureCluster
```

## Loading Data

1. Use the MNIST dataset as the training set `ds_train`. The `ds_train` contains only image data and does not contain labels.

2. Mix MNIST and CIFAR-10 into a dataset as test set `ds_test`, which contains only image data and does not contain labels.

3. Use another mixed dataset of MNIST and CIFAR-10 as a validation sample and record it as `ds_eval`. `ds_eval` contains only image data and does not contain labels. `ds_eval` is marked separately. Non-OOD samples are marked as 0, OOD samples are marked as 1, and `ds_eval` is marked as `ood_label`.

```python
ds_train = np.load('/dataset/concept_train_lenet.npy')
ds_test = np.load('/dataset/concept_test_lenet2.npy')
ds_eval = np.load('/dataset/concept_test_lenet1.npy')
```

`ds_train(numpy.ndarray)`: training set, which contains only image data.

`ds_test(numpy.ndarray)`: test set, which contains only image data.

`ds_eval(numpy.ndarray)`: validation set, which contains only image data.

## Loading a Neural Network Model

Use the training set `ds_train` and its classification `label` to train the neural network LeNet and load the model. Here, we directly import the trained model file.

The `label` here is different from the `ood_label` mentioned above. The `label` indicates the classification label of the sample, and `ood_label` indicates whether the sample belongs to the OOD label.

```python
ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
net = LeNet5()
load_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, load_dict)
model = Model(net)
```

`ckpt_path(str)`: model file path.

It should be noted that, to extract feature output of a specific layer by using a neural network, functions of the feature extraction and the naming neural layer need to be added to a neural network construction process.
The `layer` is used to name the neural network layer. You can use the following method to reconstruct the neural network model, name the neural network at each layer, and obtain the feature output value.

1. Import the `TensorSummary` module.
2. Add `self.summary = TensorSummary()` to the initialization function `__init__`.
3. Add `self.summary(`name`, x)` after the constructor function of each layer of the neural network.

In this test case, the KMeans function in sklearn is used for feature clustering analysis. Therefore, the input data dimension of KMeans must be two-dimensional. LeNet is used as an example. The features of the fully-connected layer and ReLU layer from the bottom five layers are extracted. The data dimensions meet the KMeans requirements.

The LeNet neural network construction process is as follows:

```python
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import TensorSummary

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Wrap conv."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")

def fc_with_initialize(input_channels, out_channels):
    """Wrap initialize method of full connection layer."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

def weight_variable():
    """Wrap initialize variable."""
    return TruncatedNormal(0.05)

class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.summary = TensorSummary()

    def construct(self, x):
        """
        construct the network architecture
        Returns:
            x (tensor): network output
        """
        x = self.conv1(x)

        x = self.relu(x)

        x = self.max_pool2d(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.max_pool2d(x)

        x = self.flatten(x)

        x = self.fc1(x)
        self.summary('8', x)

        x = self.relu(x)
        self.summary('9', x)

        x = self.fc2(x)
        self.summary('10', x)

        x = self.relu(x)
        self.summary('11', x)

        x = self.fc3(x)
        self.summary('output', x)
        return x

```

## Initializing the Image Concept Drift Detection Module

Import the concept drift detection module and initialize it.

```python
detector = OodDetectorFeatureCluster(model, ds_train, n_cluster=10, layer='output[:Tensor]')
```

`model(Model)`: neural network model, which is obtained by training the training set `ds_train` and its classification labels.

`ds_train(numpy.ndarray)`: training set, which contains only image data.

`n_cluster(int)`: number of feature clusters.

`layer(str)`: name of the layer used by the neural network to extract features.

Note that during `OodDetectorFeatureCluster` initialization, the `[:Tensor]` suffix needs to be added after the `layer` parameter.
For example, if a neural network layer is named `name`, then `layer='name[:Tensor]'`. In the `layer='output[:Tensor]` instance, the feature `output` of the last layer of LeNet is used, that is, `layer='output[:Tensor]`. In addition, the algorithm uses the KMeans function in sklearn to perform feature clustering analysis. The input data dimension of KMeans must be two-dimensional. Therefore, the features extracted by `layer` must be two-dimensional data, such as the fully-connected layer and ReLU layer in the LeNet example above.

## Obtaining an Optimal Concept Drift Detection Threshold

The optimal concept drift detection threshold is obtained based on the validation set `ds_eval` and its OOD label `ood_label`.

The validation set `ds_eval` can be constructed manually. For example, it consists of 50% of the MNIST dataset and 50% of the CIFAR-10 dataset. Therefore, the OOD label `ood_label` indicates that the label values of the first 50% are 0 and those of the last 50% are 1.

```python
num = int(len(ds_eval) / 2)
ood_label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)  # ID data = 0, OOD data = 1
optimal_threshold = detector.get_optimal_threshold(ood_label, ds_eval)
```

`ds_eval(numpy.ndarray)`: validation set, which contains only image data.
`ood_label(numpy.ndarray)`: OOD label of validation set `ds_eval`. Non-OOD samples are marked as 0, and OOD samples are marked as 1.

Certainly, if it is difficult for a user to obtain `ds_eval` and the OOD label `ood_label`, a value of `optimal_threshold` may be manually and flexibly set, and the value of `optimal_threshold` is a floating point number between [0, 1].

## Executing the Concept Drift Detection

```python
result = detector.ood_predict(optimal_threshold, ds_test)
```

`ds_test(numpy.ndarray)`: test set, which contains only image data.
`optimal_threshold(float)`: optimal threshold. You can obtain the values by running the `detector.get_optimal_threshold(ood_label, ds_eval)` command.
However, if it is difficult for a user to obtain `ds_eval` and the OOD label `ood_label`, a value of `optimal_threshold` may be manually and flexibly set, and the value of `optimal_threshold` is a floating point number between [0, 1].

## Viewing the Result

```python
print(result)
```

`result(numpy.ndarray)`: one-dimensional array consisting of elements 0 and 1, corresponding to the OOD detection result of `ds_test`.
For example, if `ds_test` is a dataset consisting of five MNIST datasets and five CIFAR-10 datasets, the detection result is [0,0,0,0,0,1,1,1,1,1].
