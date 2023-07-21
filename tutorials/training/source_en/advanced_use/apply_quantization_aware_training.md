# Applying Quantization Aware Training

`Linux` `Ascend` `GPU` `Model Optimization` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/apply_quantization_aware_training.md)

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
| Network       | Supports networks such as LeNet and ResNet50. For details, see <https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo>. |
| Algorithm     | Supports symmetric and asymmetric quantization algorithms in MindSpore fake quantization training. |
| Solution      | Supports 4-, 7-, and 8-bit quantization solutions. |

## Quantization Aware Training Example

The procedure for the quantization aware training model is the same as that for the common training. After the network is defined and the model is generated, additional operations need to be performed. The complete process is as follows:

1. Process data and load datasets.
2. Define an original unquantative network.
3. Define a fusion network. After defining a original unquantative network, replace the specified operators to define a fusion network.
4. Define an optimizer and loss function.
5. Generate a quantization network. Insert a fake quantization node into the fusion network by using a conversion API, a quantization network will be generated based on the fusion network.
6. Perform quantization training. Generate a quantization model based on the quantization network training.

Compared with common training, the quantization aware training requires additional steps which are steps 3, 5, and 6 in the preceding process.

> - Fusion network: network after the specified operators (`nn.Conv2dBnAct` and `nn.DenseBnAct`) are used for replacement.
> - Quantization network: network obtained after the fusion model uses a conversion API (`convert_quant_network`) to insert a fake quantization node.
> - Quantization model: model in the checkpoint format obtained after the quantization network training.

Next, the LeNet network is used as an example to describe steps 2 and 3.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo/official/cv/lenet_quant>.

### Defining a Fusion Network

Define a fusion network and replace the specified operators.

1. Use the `nn.Conv2dBnAct` operator to replace the two operators `nn.Conv2d` and `nn.ReLU` in the original network model.
2. Use the `nn.DenseBnAct` operator to replace the two operators `nn.Dense` and `nn.ReLU` in the original network model.

> Even if the `nn.Dense` and `nn.Conv2d` operators are not followed by `nn.BatchNorm*` and `nn.ReLU`, the preceding two replacement operations must be performed as required.

The definition of the original network model LeNet5 is as follows:

```python
class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.
        num_channel (int): Num channel. Default: 1.
    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10, num_channel=1)
    
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
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
```

The following shows the fusion network after operators are replaced:

```python
class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        
        self.conv1 = nn.Conv2dBnAct(1, 6, kernel_size=5,  activation='relu')
        self.conv2 = nn.Conv2dBnAct(6, 16, kernel_size=5, activation='relu')
        
        self.fc1 = nn.DenseBnAct(16 * 5 * 5, 120, activation='relu')
        self.fc2 = nn.DenseBnAct(120, 84, activation='relu')
        self.fc3 = nn.DenseBnAct(84, self.num_class)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.max_pool2d(self.conv1(x))
        x = self.max_pool2d(self.conv2(x))
        x = self.flattern(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

### Converting the Fusion Model into a Quantization Network

Use the `convert_quant_network` API to automatically insert a fake quantization node into the fusion model to convert the fusion model into a quantization network.

```python
from mindspore.train.quant import quant

net = quant.convert_quant_network(network, quant_delay=900, bn_fold=False, per_channel=[True, False], symmetric=[False, False])
```

## Retraining and Inference

### Importing a Model for Retraining

The preceding describes the quantization aware training from scratch. A more common case is that an existing model file needs to be converted to a quantization model. The model file and training script obtained through common network model training are available for quantization aware training. To use a checkpoint file for retraining, perform the following steps:

    1. Process data and load datasets.
    2. Define an original unquantative network.
    3. Train the original network to generate a unquantative model.
    4. Define a fusion network.
    5. Define an optimizer and loss function.
    6. Generate a quantative network based on the fusion network.
    7. Load a model file and retrain the model. Load the unquantative model file generated in step 3 and retrain the quantative model based on the quantative network to generate a quantative model. For details, see <https://www.mindspore.cn/tutorial/training/en/r1.0/use/load_model_for_inference_and_transfer.html>.

### Inference

The inference using a quantization model is the same the common model inference. The inference can be performed by directly using the checkpoint file or converting the checkpoint file into a common model format (such as AIR or MINDIR). 

For details, see <https://www.mindspore.cn/tutorial/inference/en/r1.0/multi_platform_inference.html>.

- To use a checkpoint file obtained after quantization aware training for inference, perform the following steps:

  1. Load the quantization model.
  2. Perform the inference.

- Convert the checkpoint file into a common model format such as ONNX for inference. (This function is coming soon.) 

## References

[1] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

[2] Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: A whitepaper[J]. arXiv preprint arXiv:1806.08342, 2018.

[3] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.
