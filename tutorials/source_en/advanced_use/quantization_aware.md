# Quantization

`Ascend` `GPU` `Model Optimization` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r0.7/tutorials/source_en/advanced_use/quantization_aware.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Background

Deep learning technologies are used on an increasing number of applications on mobile or edge devices. Take mobile phones as an example. To provide user-friendly and intelligent services, the deep learning function is integrated into operating systems and applications. However, this function involves training or inference, containing a large number of models and weight files. The original weight file of AlexNet has exceeded 200 MB, and the new model is developing towards a more complex structure with more parameters. Due to limited hardware resources of a mobile or edge device, a model needs to be simplified and the quantization technology is used to solve this problem.

## Concept

### Quantization

Quantization is a process of approximating (usually INT8) a fixed point of a floating-point model weight of a continuous value (or a large quantity of possible discrete values) or tensor data that flows through a model to a limited quantity (or a relatively small quantity) of discrete values at a relatively low inference accuracy loss. It is a process of approximately representing 32-bit floating-point data with fewer bits, while the input and output of the model are still floating-point data. In this way, the model size and memory usage can be reduced, the model inference speed can be accelerated, and the power consumption can be reduced.

As described above, compared with the FP32 type, low-accuracy data representation types such as FP16, INT8, and INT4 occupy less space. Replacing the high-accuracy data representation type with the low-accuracy data representation type can greatly reduce the storage space and transmission time. Low-bit computing has higher performance. Compared with FP32, INT8 has a three-fold or even higher acceleration ratio. For the same computing, INT8 has obvious advantages in power consumption.

Currently, there are two types of quantization solutions in the industry: quantization aware training and post-training quantization training.

### Fake Quantization Node

A fake quantization node is a node inserted during quantization aware training, and is used to search for network data distribution and feed back a lost accuracy. The specific functions are as follows:
- Find the distribution of network data, that is, find the maximum and minimum values of the parameters to be quantized.
- Simulate the accuracy loss of low-bit quantization, apply the loss to the network model, and transfer the loss to the loss function, so that the optimizer optimizes the loss value during training.

## Quantization Aware Training

MindSpore quantization aware training is to replace high-accuracy data with low-accuracy data to simplify the model training process. In this process, the accuracy loss is inevitable. Therefore, a fake quantization node is used to simulate the accuracy loss, and backward propagation learning is used to reduce the accuracy loss. MindSpore adopts the solution in reference [1] for the quantization of weights and data.

Aware quantization training specifications

| Specification | Description                              |
| ------------- | ---------------------------------------- |
| Hardware      | Supports hardware platforms based on the GPU or Ascend AI 910 processor. |
| Network       | Supports networks such as LeNet and ResNet50. For details, see <https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo>. |
| Algorithm     | Supports symmetric and asymmetric quantization algorithms in MindSpore fake quantization training. |
| Solution      | Supports 4-, 7-, and 8-bit quantization solutions. |

## Quantization Aware Training Example

The procedure for the quantization aware training model is the same as that for the common training. After the network is defined and the model is generated, additional operations need to be performed. The complete process is as follows:

1. Process data and load datasets.
2. Define a network.
3. Define a fusion network. After a network is defined, replace the specified operators to define a fusion network.
4. Define an optimizer and loss function.
5. Perform model training. Generate a fusion model based on the fusion network training.
6. Generate a quantization network. After the fusion model is obtained based on the fusion network training, insert a fake quantization node into the fusion model by using a conversion API to generate a quantization network.
7. Perform quantization training. Generate a quantization model based on the quantization network training.

Compared with common training, the quantization aware training requires additional steps which are steps 3, 6, and 7 in the preceding process.

> - Fusion network: network after the specified operators (`nn.Conv2dBnAct` and `nn.DenseBnAct`) are used for replacement.
> - Fusion model: model in the checkpoint format generated by the fusion network training.
> - Quantization network: network obtained after the fusion model uses a conversion API (`convert_quant_network`) to insert a fake quantization node.
> - Quantization model: model in the checkpoint format obtained after the quantization network training.

Next, the LeNet network is used as an example to describe steps 3 and 6.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo/official/cv/lenet_quant>.

### Defining a Fusion Network

Define a fusion network and replace the specified operators.

1. Use the `nn.Conv2dBnAct` operator to replace the two operators `nn.Conv2d` and `nn.ReLU` in the original network model.
2. Use the `nn.DenseBnAct` operator to replace the two operators `nn.Dense` and `nn.ReLU` in the original network model.

> Even if the `nn.Dense` and `nn.Conv2d` operators are not followed by `nn.BatchNorm*` and `nn.ReLU`, the preceding two replacement operations must be performed as required.

The definition of the original network model LeNet5 is as follows:

```python
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """
    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
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
```

The following shows the fusion network after operators are replaced:

```python
class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        
        self.conv1 = nn.Conv2dBnAct(1, 6, kernel_size=5, batchnorm=True, activation='relu')
        self.conv2 = nn.Conv2dBnAct(6, 16, kernel_size=5, batchnorm=True, activation='relu')
        
        self.fc1 = nn.DenseBnAct(16 * 5 * 5, 120, activation='relu')
        self.fc2 = nn.DenseBnAct(120, 84, activation='relu')
        self.fc3 = nn.DenseBnAct(84, self.num_class)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.max_pool2d(x)
        x = self.flattern(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

### Converting the Fusion Model into a Quantization Network

Use the `convert_quant_network` API to automatically insert a fake quantization node into the fusion model to convert the fusion model into a quantization network.

```python
from mindspore.train.quant import quant as qat

net = qat.convert_quant_network(net, quant_delay=0, bn_fold=False, freeze_bn=10000, weight_bits=8, act_bits=8)
```

## Retraining and Inference

### Importing a Model for Retraining

The preceding describes the quantization aware training from scratch. A more common case is that an existing model file needs to be converted to a quantization model. The model file and training script obtained through common network model training are available for quantization aware training. To use a checkpoint file for retraining, perform the following steps:

    1. Process data and load datasets.
    2. Define a network.
    3. Define a fusion network.
    4. Define an optimizer and loss function.
    5. Load a model file and retrain the model. Load an existing model file and retrain the model based on the fusion network to generate a fusion model. For details, see <https://www.mindspore.cn/tutorial/en/r0.7/use/saving_and_loading_model_parameters.html#id6>.
    6. Generate a quantization network.
    7. Perform quantization training.

### Inference

The inference using a quantization model is the same as common model inference. The inference can be performed by directly using the checkpoint file or converting the checkpoint file into a common model format (such as ONNX or AIR). 

For details, see <https://www.mindspore.cn/tutorial/en/r0.7/use/multi_platform_inference.html>.

- To use a checkpoint file obtained after quantization aware training for inference, perform the following steps:

  1. Load the quantization model.
  2. Perform the inference.

- Convert the checkpoint file into a common model format such as ONNX for inference. (This function is coming soon.) 

## References

[1] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

[2] Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: A whitepaper[J]. arXiv preprint arXiv:1806.08342, 2018.

[3] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.
