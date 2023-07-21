# 量化

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_zh_cn/advanced_use/aware_quantization.md)

## 背景

越来越多的应用选择在移动设备或者边缘设备上使用深度学习技术。以手机为例，为了提供人性化和智能的服务，现在操作系统和应用都开始集成深度学习功能。而使用该功能，涉及训练或者推理，自然包含大量的模型及权重文件。经典的AlexNet，原始权重文件已经超过了200MB，而最近出现的新模型正往结构更复杂、参数更多的方向发展。由于移动设备、边缘设备的硬件资源有限，需要对模型进行精简，而量化（Quantization）技术就是应对该类问题衍生出的技术之一。

## 概念

### 量化

量化即以较低的推理精度损失将连续取值（或者大量可能的离散取值）的浮点型模型权重或流经模型的张量数据定点近似（通常为INT8）为有限多个（或较少的）离散值的过程，它是以更少位数的数据类型用于近似表示32位有限范围浮点型数据的过程，而模型的输入输出依然是浮点型。这样的好处是可以减小模型尺寸大小，减少模型内存占用，加快模型推理速度，降低功耗等。

如上所述，与FP32类型相比，FP16、INT8、INT4等低精度数据表达类型所占用空间更小。使用低精度数据表达类型替换高精度数据表达类型，可以大幅降低存储空间和传输时间。而低比特的计算性能也更高，INT8相对比FP32的加速比可达到3倍甚至更高，对于相同的计算，功耗上也有明显优势。

当前业界量化方案主要分为两种：感知量化训练（Aware Quantization Training）和训练后量化（Post-training Quantization）。

### 伪量化节点

伪量化节点，是指感知量化训练中插入的节点，用以寻找网络数据分布，并反馈损失精度，具体作用如下：
- 找到网络数据的分布，即找到待量化参数的最大值和最小值；
- 模拟量化为低比特时的精度损失，把该损失作用到网络模型中，传递给损失函数，让优化器在训练过程中对该损失值进行优化。

## 感知量化训练

MindSpore的感知量化训练是在训练基础上，使用低精度数据替换高精度数据来简化训练模型的过程。这个过程不可避免引入精度的损失，这时使用伪量化节点来模拟引入的精度损失，并通过反向传播学习，来减少精度损失。对于权值和数据的量化，MindSpore采用了参考文献[1]中的方案。

感知量化训练规格

| 规格 | 规格说明 |
| --- | --- |
| 硬件支持 | GPU、Ascend AI 910处理器的硬件平台 |
| 网络支持 | 已实现的网络包括LeNet、ResNet50等网络，具体请参见<https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo>。 |
| 算法支持 | 在MindSpore的伪量化训练中，支持非对称和对称的量化算法。 |
| 方案支持 | 支持4、7和8比特的量化方案。 |

## 感知量化训练示例

感知量化训练模型与一般训练步骤一致，在定义网络和最后生成模型阶段后，需要进行额外的操作，完整流程如下：

1.  数据处理加载数据集。
2.  定义网络。
3.  定义融合网络。在完成定义网络后，替换指定的算子，完成融合网络的定义。
4.  定义优化器和损失函数。
5.  进行模型训练。基于融合网络训练生成融合模型。
6.  转化量化网络。基于融合网络训练后得到的融合模型，使用转化接口在融合模型中插入伪量化节点，生成的量化网络。
7.  进行量化训练。基于量化网络训练，生成量化模型。

在上面流程中，第3、6、7步是感知量化训练区别普通训练需要额外进行的步骤。

> - 融合网络：使用指定算子（`nn.Conv2dBnAct`、`nn.DenseBnAct`）替换后的网络。
> - 融合模型：使用融合网络训练生成的checkpoint格式的模型。
> - 量化网络：融合模型使用转换接口（`convert_quant_network`）插入伪量化节点后得到的网络。
> - 量化模型：量化网络训练后得到的checkpoint格式的模型。

接下来，以LeNet网络为例，展开叙述3、6两个步骤。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo/lenet_quant> 。

### 定义融合网络

定义融合网络，在定义网络后，替换指定的算子。

1. 使用`nn.Conv2dBnAct`算子替换原网络模型中的3个算子`nn.Conv2d`、`nn.batchnorm`和`nn.relu`。
2. 使用`nn.DenseBnAct`算子替换原网络模型中的3个算子`nn.Dense`、`nn.batchnorm`和`nn.relu`。

> 即使`nn.Dense`和`nn.Conv2d`算子后面没有`nn.batchnorm`和`nn.relu`，都要按规定使用上述两个算子进行融合替换。

原网络模型的定义如下所示：

```python
class LeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.batchnorm(6)
        self.act1 = nn.relu()
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.batchnorm(16)
        self.act2 = nn.relu()
        
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.act3 = nn.relu()
        self.fc3 = nn.Dense(84, self.num_class)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.max_pool2d(x)
        x = self.flattern(x)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act3(x)
        x = self.fc3(x)
        return x
```

替换算子后的融合网络如下：

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

### 转化为量化网络

使用`convert_quant_network`接口自动在融合模型中插入伪量化节点，将融合模型转化为量化网络。

```python
from mindspore.train.quant import quant as qat

net = qat.convert_quant_network(net, quant_delay=0, bn_fold=False, freeze_bn=10000, weight_bits=8, act_bits=8)
```

## 重训和推理

### 导入模型重新训练

上面介绍了从零开始进行感知量化训练。更常见情况是已有一个模型文件，希望生成量化模型，这时已有正常网络模型训练得到的模型文件及训练脚本，进行感知量化训练。这里使用checkpoint文件重新训练的功能，详细步骤为：

  1.  数据处理加载数据集。
  2.  定义网络。
  3.  定义融合网络。
  4.  定义优化器和损失函数。
  5.  加载模型文件模型重训。加载已有模型文件，基于融合网络重新训练生成融合模型。详细模型重载训练，请参见<https://www.mindspore.cn/tutorial/zh-CN/r0.5/use/saving_and_loading_model_parameters.html#id6>
  6.  转化量化网络。
  7.  进行量化训练。

### 进行推理

使用量化模型进行推理，与普通模型推理一致，分为直接checkpoint文件推理及转化为通用模型格式（ONNX、GEIR等）进行推理。

> 推理详细说明请参见<https://www.mindspore.cn/tutorial/zh-CN/r0.5/use/multi_platform_inference.html>。

- 使用感知量化训练后得到的checkpoint文件进行推理：

  1.  加载量化模型。
  2.  推理。

- 转化为ONNX等通用格式进行推理（暂不支持，开发完善后补充）。
   
## 参考文献

[1] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

[2] Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: A whitepaper[J]. arXiv preprint arXiv:1806.08342, 2018.

[3] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.
