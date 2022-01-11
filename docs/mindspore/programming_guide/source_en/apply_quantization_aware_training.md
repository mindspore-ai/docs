# Applying Quantization Aware Training

`Ascend` `GPU` `Function Extension`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_en/apply_quantization_aware_training.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Background

Deep learning technologies are used on an increasing number of applications on mobile or edge devices. Take mobile phones as an example. To provide user-friendly and intelligent services, the deep learning function is integrated into operating systems and applications. However, this function involves training or inference, containing a large number of models and weight files. The original weight file of AlexNet has exceeded 200 MB, and the new model is developing towards a more complex structure with more parameters. Due to limited hardware resources of a mobile or edge device, a model needs to be simplified and the quantization technology is used to solve this problem.

## Concepts

### Quantization

Quantization is a process in which weights of a floating-point model with continuous values or tensor data flowing through the model are approximated at fixed points (usually INT8) to a limited quantity (or a relatively small quantity) of discrete values at a low inference precision loss. It is a process of approximately representing 32-bit floating-point data with fewer bits, while the input and output of the model are still floating-point data. In this way, the model size and memory usage can be reduced, the model inference speed can be accelerated, and the power consumption can be reduced.

As described above, compared with the FP32 type, low-accuracy data representation types such as FP16, INT8, and INT4 occupy less space. Replacing the high-accuracy data representation type with the low-accuracy data representation type can greatly reduce the storage space and transmission time. Low-bit computing has higher performance. Compared with FP32, INT8 has a three-fold or even higher acceleration ratio. For the same computing, INT8 has obvious advantages in power consumption.

Currently, there are two types of quantization solutions in the industry: quantization aware training and post-training quantization. Quantization aware training requires training data and generally has better performance in model accuracy. It is applicable to scenarios that have high requirements on the model compression rate and model accuracy. Post-training quantization is easy to use. Only a small amount of calibration data is required. This mode applies to scenarios that require high usability and lack training resources.

### Fake Quantization Node

A fake quantization node is a node inserted during quantization aware training, and is used to search for network data distribution and feed back a lost accuracy. The specific functions are as follows:

- Find the distribution of network data, that is, find the maximum and minimum values of the parameters to be quantized.
- Simulate the accuracy loss of low-bit quantization, apply the loss to the network model, and transfer the loss to the loss function, so that the optimizer optimizes the loss value during training.

## Quantization Aware Training

MindSpore's quantization aware training uses fake quantization nodes to simulate quantization operations. During the training, floating-point numbers are still used for computation, and network parameters are updated through backward propagation learning, so that the network parameters can better adapt to the loss caused by quantization. MindSpore adopts the solution in reference [1] for the quantization of weights and data.

Aware quantization training specifications

| Specification | Description |
| --- | --- |
| Hardware | Supports hardware platforms based on the GPU or Ascend AI 910 processor. |
| Network | Supports networks such as LeNet and ResNet50. For details, see <https://gitee.com/mindspore/models/tree/master>.  |
| Algorithm | Supports asymmetric and symmetric quantization algorithms, as well as layer-by-layer and channel-by-channel quantization algorithms. |
| Solution | Supports 4-, 7-, and 8-bit quantization solutions.  |
| Data Type | Supports the FP32 and FP16 networks for quantization training on Ascend, and the FP32 network on GPU.  |
| Running Mode | Supports graph mode. |

## Quantization Aware Training Example

The procedure of quantization aware training is the same as that of common training. Additional operations need to be performed in the phases of defining a quantization network and generating a quantization model. The complete process is as follows:

1. Load the dataset and process data.
2. Define a quantization network.
3. Define an optimizer and a loss function.
4. Train the network and save the model file.
5. Load the saved model for inference.
6. Export a quantization model.

Compared with common training, the quantization aware training requires additional steps which are steps 2 and 6 in the preceding process. Next, the LeNet network is used as an example to describe quantization-related steps.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/models/tree/r1.6/official/cv/lenet_quant>.

### Defining a Quantization Network

A quantization network is a network with fake quantization nodes generated after the network layer to be quantized is modified based on the original network definition. There are two methods for defining a quantization network:

- Automatically build a quantization network: After a fusion network is defined and the conversion API is called, the fusion network is automatically converted into a quantization network. You do not need to be aware of the process of inserting fake quantization nodes.
- Manually build a quantization network: You need to manually replace a network layer to be quantized with a corresponding quantization node, or directly insert a fake quantization node behind the network layer to be quantized. The modified network is a quantization network. You can customize the network layer to be quantized, which is more flexible and easy to scale.

> - The automatically building method supports the quantization of the following network layers: `nn.Conv2dBnAct`, `nn.DenseBnAct`, `Add`, `Sub`, `Mul`, and `RealDiv`. If only some of these network layers need to be quantized or other network layers need to be quantized, use the manually building method.
> - The conversion API for automatically building is `QuantizationAwareTraining.quantize`.

The original network model LeNet5 is defined as follows:

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

#### Automatically Building a Quantization Network

First, define a fusion network:

1. Use the `nn.Conv2dBnAct` operator to replace the two operators `nn.Conv2d` and `nn.ReLU` in the original network model.
2. Use the `nn.DenseBnAct` operator to replace the two operators `nn.Dense` and `nn.ReLU` in the original network model.

> Even if the `nn.Dense` and `nn.Conv2d` operators are not followed by `nn.BatchNorm` and `nn.ReLU`, the preceding two replacement operations must be performed as required.

The following shows the fusion network after operators are replaced:

```python
class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class

        self.conv1 = nn.Conv2dBnAct(1, 6, kernel_size=5, pad_mode='valid', activation='relu')
        self.conv2 = nn.Conv2dBnAct(6, 16, kernel_size=5, pad_mode='valid', activation='relu')

        self.fc1 = nn.DenseBnAct(16 * 5 * 5, 120, activation='relu')
        self.fc2 = nn.DenseBnAct(120, 84, activation='relu')
        self.fc3 = nn.DenseBnAct(84, self.num_class)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.conv1(x))
        x = self.max_pool2d(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

When the quantization aware training is used for fine-tuning, the parameters of the pre-trained model need to be loaded.

```python
from mindspore.compression.quant import load_nonquant_param_into_quant_net
...
# define fusion network
network = LeNet5(cfg.num_classes)

param_dict = load_checkpoint(args.ckpt_path)
load_nonquant_param_into_quant_net(network, param_dict)
```

Use the `QuantizationAwareTraining.quantize` API to automatically insert a fake quantization node into the fusion network to convert the fusion network into a quantization network.

```python
from mindspore.compression.quant import QuantizationAwareTraining

quantizer = QuantizationAwareTraining(quant_delay=900,
                                      bn_fold=False,
                                      per_channel=[True, False],
                                      symmetric=[True, False])
net = quantizer.quantize(network)
```

> If the quantization precision does not meet the requirement, adjust the quantization policy parameters. For example, generally, a larger quantity of quantization bits results in a smaller precision loss, and channel-by-channel quantization provides greater precision than layer-by-layer quantization. In addition, you can manually build a quantization network. Select some of the network layers to be quantized to balance the relationship between accuracy and inference performance.

#### Manually Building a Quantization Network

Replace the layers that need to be quantized in the original network with the corresponding quantization operators:

1. Use `nn.Conv2dQuant` to replace the `nn.Conv2d` operator in the original network model.
2. Use `nn.DenseQuant` to replace the `nn.Dense` operator in the original network model.
3. Use `nn.ActQuant` to replace the `nn.ReLU` operator in the original network model.

```python
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class

        self.qconfig = create_quant_config(quant_dtype=(QuantDtype.INT8, QuantDtype.INT8), per_channel=(True, False), symmetric=[True, False])

        self.conv1 = nn.Conv2dQuant(channel, 6, 5, pad_mode='valid', quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.conv2 = nn.Conv2dQuant(6, 16, 5, pad_mode='valid', quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.fc1 = nn.DenseQuant(16 * 5 * 5, 120, quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.fc2 = nn.DenseQuant(120, 84, quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
        self.fc3 = nn.DenseQuant(84, self.num_class, quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)

        self.relu = nn.ActQuant(nn.ReLU(), quant_config=self.qconfig, quant_dtype=QuantDtype.INT8)
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

> - Quantization operators: `nn.Conv2dQuant`, `nn.DenseQuant` and `nn.ActQuant` are operators that contain fake quantization nodes. For details about quantization operators, see <https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.nn.html#quantized-functions>.
> - The fake quantization node `nn.FakeQuantWithMinMaxObserver` can be inserted after the network layer that needs to be quantized to implement quantization of more network layers.
> - You are advised to preferentially select the layer at the rear of the quantization network because the network layer at the front of the quantization network may cause more precision loss.

When the quantization aware training is used for fine-tuning, the parameters of the pre-trained model need to be loaded.

```python
from mindspore.compression.quant import load_nonquant_param_into_quant_net
...
# define quant network
network = LeNet5(cfg.num_classes)

param_dict = load_checkpoint(args.ckpt_path)
load_nonquant_param_into_quant_net(network, param_dict)
```

### Exporting a Quantization Model

The quantization model deployed on the device hardware platform is in a general model format (such as AIR and MindIR), and does not include a fake quantization node. The export procedure is as follows:

1. Define a quantization network. A quantization network in this step is the same as a quantization network in quantization aware training.
2. Load the checkpoint file saved during quantization aware training.
3. Export a quantization model. Set the `quant_mode`, `mean` and `std_dev` parameter of the `export` API.

```python
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

if __name__ == "__main__":
    ...
    # define fusion network
    network = LeNet5(cfg.num_classes)
    quantizer = QuantizationAwareTraining(bn_fold=False,
                                          per_channel=[True, False],
                                          symmetric=[True, False])
    network = quantizer.quantize(network)  

    # load quantization aware network checkpoint
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.ones([1, 1, cfg.image_height, cfg.image_width]), mindspore.float32)
    export(network, inputs, file_name="lenet_quant", file_format='MINDIR', quant_mode='QUANT', mean=127.5, std_dev=127.5)
```

After the quantization model is exported, use MindSpore for inference. For details, see [Inference Using MindSpore](https://www.mindspore.cn/docs/programming_guide/en/r1.6/index.html).

> - The exported model can be in MindIR or AIR format.
> - Models exported after quantization aware training support [Inference on Devices](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/index.html) and [Inference on Ascend 310](https://www.mindspore.cn/docs/programming_guide/en/r1.6/multi_platform_inference_ascend_310.html).

## References

[1] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

[2] Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: A whitepaper[J]. arXiv preprint arXiv:1806.08342, 2018.
