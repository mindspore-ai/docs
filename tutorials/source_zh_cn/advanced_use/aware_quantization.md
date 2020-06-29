# 量化

<!-- TOC -->

- [量化](#量化)
    - [概述](#概述)
    - [感知量化训练](#感知量化训练)
        - [伪量化节点](#伪量化节点)
        - [感知量化示例](#感知量化示例)
        - [导入模型重训与推理](#导入模型重训与推理)
    - [参考文献](#参考文献)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_zh_cn/advanced_use/aware_quantization.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

与FP32类型相比，FP16、INT8、INT4的低精度数据表达类型所占用空间更小，因此对应的存储空间和传输时间都可以大幅下降。以手机为例，为了提供更人性化和智能的服务，现在越来越多的OS和APP集成了深度学习的功能，自然需要包含大量的模型及权重文件。经典的AlexNet，原始权重文件的大小已经超过了200MB，而最近出现的新模型正在往结构更复杂、参数更多的方向发展。显然，低精度类型的空间受益还是很明显的。低比特的计算性能也更高，INT8相对比FP32的加速比可达到3倍甚至更高，功耗上也对应有所减少。

量化即以较低的推理精度损失将连续取值（或者大量可能的离散取值）的浮点型模型权重或流经模型的张量数据定点近似（通常为int8）为有限多个（或较少的）离散值的过程，它是以更少位数的数据类型用于近似表示32位有限范围浮点型数据的过程，而模型的输入输出依然是浮点型，从而达到减少模型尺寸大小、减少模型内存消耗及加快模型推理速度等目标。

量化方案主要分为两种：感知量化训练（aware quantization training）和训练后量化（post-training quantization）。

## 感知量化训练

感知量化训练为在网络模型训练的过程中，插入伪量化节点进行伪量化训练的过程。

MindSpore的感知量化训练是一种伪量化的过程，它是在可识别的某些操作内嵌入伪量化节点，用以统计训练时流经该节点数据的最大最小值。其目的是减少精度损失，其参与模型训练的前向推理过程令模型获得量化损失，但梯度更新需要在浮点下进行，因而其并不参与反向传播过程。

在MindSpore的伪量化训练中，支持非对称和对称的量化算法，支持4、7和8bit的量化方案。

目前MindSpore感知量化训练支持的后端有GPU和Ascend。

### 伪量化节点

伪量化节点的作用：（1）找到网络数据的分布，即找到待量化参数的最大值和最小值；（2）模拟量化到低比特操作的时候的精度损失，把该损失作用到网络模型中，传递给损失函数，让优化器去在训练过程中对该损失值进行优化。

伪量化节点的意义在于统计流经数据的min和max值，并参与前向传播，让损失函数的值增大，优化器感知到损失值的增加，并进行持续性地反向传播学习，进一步减少因为伪量化操作而引起的精度下降，从而提升精确度。

对于权值和数据的量化，MindSpore都采用参考文献[1]中的方案进行量化。

### 感知量化示例

使用感知量化训练特性，主要的步骤为：

1.  定义网络模型
2.  量化自动构图

代码样例如下：
    

1. 定义网络模型
 
    以LeNet5网络模型为例子，原网络模型的定义如下所示。

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

    融合网络模型定义：
    
    使用`nn.Conv2dBnAct`算子替换原网络模型中的三个算子`nn.Conv2d`、`nn.batchnorm`和`nn.relu`；
    
    同理，使用`nn.DenseBnAct`算子替换原网络模型中的对应的算子`nn.Dense`、`nn.batchnorm`和`nn.relu`。
    
    即使`nn.Dense`和`nn.Conv2d`算子后面没有`nn.batchnorm`和`nn.relu`，都要按规定使用上述两个算子进行融合替换。

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
2. 量化自动构图

    使用`create_training_network`接口封装网络模型，该步骤将会自动在融合网络模型中插入伪量化算子。

    ```python
    from mindspore.train.quant import quant as qat
    net = qat.convert_quant_network(net, quant_delay=0, bn_fold=False, freeze_bn=10000, weight_bits=8, act_bits=8)
    ```

    其余步骤（如定义损失函数、优化器、超参数和训练网络等）与普通网络训练相同。

### 导入模型重训与推理

经过`create_training_network`函数之后，融合网络模型的图自动转换为感知量化的图。在训练和推理的时候分为下面三种情况：

- 使用融合网络模型训练得到的checkpoint文件导入，进行感知量化训练，步骤为：a)定义融合网络模型，b)加载checkpoint文件，c)转换量化自动构图，d)训练。
- 使用感知量化训练得到的checkpoint文件导入，进行感知量化推理，步骤为：a)定义融合网络模型，b)转换量化自动构图，c)加载checkpoint文件，d)推理。
- 使用正常网络模型训练得到的checkpoint文件导入，进行感知量化训练，步骤为：a)定义融合网络模型，b)加载checkpoint文件，c)训练并保存为融合网络模型对应的checkpoint文件，d)使用融合网络模型训练得到的checkpoint文件导入，进行感知量化训练。

## 参考文献

[1] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

[2] Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: A whitepaper[J]. arXiv preprint arXiv:1806.08342, 2018.

[3] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 2704-2713.

