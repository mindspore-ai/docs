# 深度概率编程

`Ascend` `GPU` `全流程` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_zh_cn/advanced_use/apply_deep_probability_programming.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

深度学习模型具有强大的拟合能力，而贝叶斯理论具有很好的可解释能力。MindSpore深度概率编程（MindSpore Deep Probabilistic Programming, MDP）将深度学习和贝叶斯学习结合，通过设置网络权重为分布、引入隐空间分布等，可以对分布进行采样前向传播，由此引入了不确定性，从而增强了模型的鲁棒性和可解释性。MDP不仅包含通用、专业的概率学习编程语言，适用于“专业”用户，而且支持使用开发深度学习模型的逻辑进行概率编程，让初学者轻松上手；此外，还提供深度概率学习的工具箱，拓展贝叶斯应用功能。

本章将详细介绍深度概率编程在MindSpore上的应用。在动手进行实践之前，确保，你已经正确安装了MindSpore 0.7.0-beta及其以上版本。本章的具体内容如下：

1. 介绍如何使用[bnn_layers模块](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/bnn_layers)实现贝叶斯神经网络（Bayesian Neural Network, BNN）；
2. 介绍如何使用[variational模块](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/infer/variational)和[dpn模块](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/dpn)实现变分自编码器（Variational AutoEncoder, VAE）；
3. 介绍如何使用[transforms模块](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/transforms)实现DNN（Deep Neural Network, DNN）一键转BNN；
4. 介绍如何使用[toolbox模块](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/nn/probability/toolbox/uncertainty_evaluation.py)实现不确定性估计。

## 使用贝叶斯神经网络

贝叶斯神经网络是由概率模型和神经网络组成的基本模型，它的权重不再是一个确定的值，而是一个分布。本例介绍了如何使用MDP中的bnn_layers模块实现贝叶斯神经网络，并利用贝叶斯神经网络实现一个简单的图片分类功能，整体流程如下：

1. 处理MNIST数据集；
2. 定义贝叶斯LeNet网络；
3. 定义损失函数和优化器；
4. 加载数据集并进行训练。

> 本例面向GPU或Ascend 910 AI处理器平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/bnn_layers>

### 处理数据集

本例子使用的是MNIST数据集，数据处理过程与教程中的[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/quick_start/quick_start.html)一致。

### 定义贝叶斯神经网络

本例使用的是Bayesian LeNet。利用bnn_layers模块构建贝叶斯神经网络的方法与构建普通的神经网络相同。值得注意的是，`bnn_layers`和普通的神经网络层可以互相组合。

```python
import mindspore.nn as nn
from mindspore.nn.probability import bnn_layers
import mindspore.ops as ops

class BNNLeNet5(nn.Cell):
    """
    bayesian Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> BNNLeNet5(num_class=10)

    """
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
        self.reshape = ops.Reshape()

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

### 定义损失函数和优化器

接下来需要定义损失函数（Loss）和优化器（Optimizer）。损失函数是深度学习的训练目标，也叫目标函数，可以理解为神经网络的输出（Logits）和标签(Labels)之间的距离，是一个标量数据。

常见的损失函数包括均方误差、L2损失、Hinge损失、交叉熵等等。图像分类应用通常采用交叉熵损失（CrossEntropy）。

优化器用于神经网络求解（训练）。由于神经网络参数规模庞大，无法直接求解，因而深度学习中采用随机梯度下降算法（SGD）及其改进算法进行求解。MindSpore封装了常见的优化器，如`SGD`、`Adam`、`Momemtum`等等。本例采用`Adam`优化器，通常需要设定两个参数，学习率（`learning_rate`）和权重衰减项（`weight_decay`）。

MindSpore中定义损失函数和优化器的代码样例如下：

```python
# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
```

### 训练网络

贝叶斯神经网络的训练过程与DNN基本相同，唯一不同的是将`WithLossCell`替换为适用于BNN的`WithBNNLossCell`。除了`backbone`和`loss_fn`两个参数之外，`WithBNNLossCell`增加了`dnn_factor`和`bnn_factor`两个参数。`dnn_factor`是由损失函数计算得到的网络整体损失的系数，`bnn_factor`是每个贝叶斯层的KL散度的系数，这两个参数是用来平衡网络整体损失和贝叶斯层的KL散度的，防止KL散度的值过大掩盖了网络整体损失。

```python
net_with_loss = bnn_layers.WithBNNLossCell(network, criterion, dnn_factor=60000, bnn_factor=0.000001)
train_bnn_network = TrainOneStepCell(net_with_loss, optimizer)
train_bnn_network.set_train()

train_set = create_dataset('./mnist_data/train', 64, 1)
test_set = create_dataset('./mnist_data/test', 64, 1)

epoch = 10

for i in range(epoch):
    train_loss, train_acc = train_model(train_bnn_network, network, train_set)

    valid_acc = validate_model(network, test_set)

    print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tvalidation Accuracy: {:.4f}'.
          format(i, train_loss, train_acc, valid_acc))
```

其中，`train_model`和`validate_model`在MindSpore中的代码样例如下：

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

## 使用变分自编码器

接下来介绍如何使用MDP中的variational模块和dpn模块实现变分自编码器。变分自编码器是经典的应用了变分推断的深度概率模型，用来学习潜在变量的表示，通过该模型，不仅可以压缩输入数据，还可以生成该类型的新图像。本例的整体流程如下：

1. 定义变分自编码器；
2. 定义损失函数和优化器；
3. 处理数据；
4. 训练网络；
5. 生成新样本或重构输入样本。

> 本例面向GPU或Ascend 910 AI处理器平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/dpn>

### 定义变分自编码器

使用dpn模块来构造变分自编码器尤为简单，你只需要自定义编码器和解码器（DNN模型），调用`VAE`接口即可。

```python
class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024, 800)
        self.fc2 = nn.Dense(800, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder()
decoder = Decoder()
vae = VAE(encoder, decoder, hidden_size=400, latent_size=20)
```

### 定义损失函数和优化器

接下来需要定义损失函数（Loss）和优化器（Optimizer）。本例使用的损失函数是`ELBO`，`ELBO`是变分推断专用的损失函数；本例使用的优化器是`Adam`。
MindSpore中定义损失函数和优化器的代码样例如下：

```python
# loss function definition
net_loss = ELBO(latent_prior='Normal', output_prior='Normal')

# optimization definition
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)

net_with_loss = nn.WithLossCell(vae, net_loss)
```

### 处理数据

本例使用的是MNIST数据集，数据处理过程与教程中的[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/quick_start/quick_start.html)一致。

### 训练网络

使用variational模块中的`SVI`接口对VAE网络进行训练。

```python
from mindspore.nn.probability.infer import SVI

vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
vae = vi.run(train_dataset=ds_train, epochs=10)
trained_loss = vi.get_train_loss()
```

通过`vi.run`可以得到训练好的网络，使用`vi.get_train_loss`可以得到训练之后的损失。

### 生成新样本或重构输入样本

利用训练好的VAE网络，我们可以生成新的样本或重构输入样本。

```python
IMAGE_SHAPE = (-1, 1, 32, 32)
generated_sample = vae.generate_sample(64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'].asnumpy(), dtype=mstype.float32)
    reconstructed_sample = vae.reconstruct_sample(sample_x)
```

## DNN一键转换成BNN

对于不熟悉贝叶斯模型的DNN研究人员，MDP提供了高级API`TransformToBNN`，支持DNN模型一键转换成BNN模型。目前在LeNet，ResNet，MobileNet，VGG等模型上验证了API的通用性。本例将会介绍如何使用transforms模块中的`TransformToBNN`API实现DNN一键转换成BNN，整体流程如下：

1. 定义DNN模型；
2. 定义损失函数和优化器；
3. 实现功能一：转换整个模型；
4. 实现功能二：转换指定类型的层。

> 本例面向GPU或Ascend 910 AI处理器平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/transforms>

### 定义DNN模型

本例使用的DNN模型是LeNet。

```python
from mindspore.common.initializer import TruncatedNormal
import mindspore.nn as nn
import mindspore.ops as ops

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
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.reshape = ops.Reshape()

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

LeNet的网络结构如下：

```text
LeNet5
  (conv1) Conv2dinput_channels=1, output_channels=6, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (conv2) Conv2dinput_channels=6, output_channels=16, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (fc1) Densein_channels=400, out_channels=120, weight=Parameter (name=fc1.weight), has_bias=True, bias=Parameter (name=fc1.bias)
  (fc2) Densein_channels=120, out_channels=84, weight=Parameter (name=fc2.weight), has_bias=True, bias=Parameter (name=fc2.bias)
  (fc3) Densein_channels=84, out_channels=10, weight=Parameter (name=fc3.weight), has_bias=True, bias=Parameter (name=fc3.bias)
  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
```

### 定义损失函数和优化器

接下来需要定义损失函数（Loss）和优化器（Optimizer）。本例使用交叉熵损失作为损失函数，`Adam`作为优化器。

```python
network = LeNet5()

# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)

net_with_loss = WithLossCell(network, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)
```

### 实例化TransformToBNN

`TransformToBNN`的`__init__`函数定义如下：

```python
class TransformToBNN:
    def __init__(self, trainable_dnn, dnn_factor=1, bnn_factor=1):
        net_with_loss = trainable_dnn.network
        self.optimizer = trainable_dnn.optimizer
        self.backbone = net_with_loss.backbone_network
        self.loss_fn = getattr(net_with_loss, "_loss_fn")
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.bnn_loss_file = None
```

参数`trainable_bnn`是经过`TrainOneStepCell`包装的可训练DNN模型，`dnn_factor`和`bnn_factor`分别为由损失函数计算得到的网络整体损失的系数和每个贝叶斯层的KL散度的系数。
MindSpore中实例化`TransformToBNN`的代码如下：

```python
from mindspore.nn.probability import transforms

bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
```

### 实现功能一：转换整个模型

`transform_to_bnn_model`方法可以将整个DNN模型转换为BNN模型。其定义如下:

```python
    def transform_to_bnn_model(self,
                               get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias,
                                                          "out_channels": dp.out_channels, "activation": dp.activation},
                               get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels,
                                                         "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size,
                                                         "stride": dp.stride, "has_bias": dp.has_bias,
                                                         "padding": dp.padding, "dilation": dp.dilation,
                                                         "group": dp.group},
                               add_dense_args=None,
                               add_conv_args=None):
        r"""
        Transform the whole DNN model to BNN model, and wrap BNN model by TrainOneStepCell.

        Args:
            get_dense_args (function): The arguments gotten from the DNN full connection layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "has_bias": dp.has_bias}.
            get_conv_args (function): The arguments gotten from the DNN convolutional layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode,
                "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias}.
            add_dense_args (dict): The new arguments added to BNN full connection layer. Default: {}.
            add_conv_args (dict): The new arguments added to BNN convolutional layer. Default: {}.

        Returns:
            Cell, a trainable BNN model wrapped by TrainOneStepCell.
       """
```

参数`get_dense_args`指定从DNN模型的全连接层中获取哪些参数，`get_conv_args`指定从DNN模型的卷积层中获取哪些参数，参数`add_dense_args`和`add_conv_args`分别指定了要为BNN层指定哪些新的参数值。需要注意的是，`add_dense_args`中的参数不能与`get_dense_args`重复，`add_conv_args`和`get_conv_args`也是如此。

在MindSpore中将整个DNN模型转换成BNN模型的代码如下：

```python
train_bnn_network = bnn_transformer.transform_to_bnn_model()
```

整个模型转换后的结构如下：

```text
LeNet5
  (conv1) ConvReparam
    in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, weight_mean=Parameter (name=conv1.weight_posterior.mean), weight_std=Parameter (name=conv1.weight_posterior.untransformed_std), has_bias=False
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (conv2) ConvReparam
    in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, weight_mean=Parameter (name=conv2.weight_posterior.mean), weight_std=Parameter (name=conv2.weight_posterior.untransformed_std), has_bias=False
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc1) DenseReparam
    in_channels=400, out_channels=120, weight_mean=Parameter (name=fc1.weight_posterior.mean), weight_std=Parameter (name=fc1.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc1.bias_posterior.mean), bias_std=Parameter (name=fc1.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc2) DenseReparam
    in_channels=120, out_channels=84, weight_mean=Parameter (name=fc2.weight_posterior.mean), weight_std=Parameter (name=fc2.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc2.bias_posterior.mean), bias_std=Parameter (name=fc2.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc3) DenseReparam
    in_channels=84, out_channels=10, weight_mean=Parameter (name=fc3.weight_posterior.mean), weight_std=Parameter (name=fc3.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc3.bias_posterior.mean), bias_std=Parameter (name=fc3.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
```

可以看到，整个LeNet网络中的卷积层和全连接层都转变成了相应的贝叶斯层。

### 实现功能二：转换指定类型的层

`transform_to_bnn_layer`方法可以将DNN模型中指定类型的层（nn.Dense或者nn.Conv2d）转换为对应的贝叶斯层。其定义如下:

```python
 def transform_to_bnn_layer(self, dnn_layer, bnn_layer, get_args=None, add_args=None):
        r"""
        Transform a specific type of layers in DNN model to corresponding BNN layer.

        Args:
            dnn_layer_type (Cell): The type of DNN layer to be transformed to BNN layer. The optional values are
            nn.Dense, nn.Conv2d.
            bnn_layer_type (Cell): The type of BNN layer to be transformed to. The optional values are
                DenseReparameterization, ConvReparameterization.
            get_args (dict): The arguments gotten from the DNN layer. Default: None.
            add_args (dict): The new arguments added to BNN layer. Default: None.

        Returns:
            Cell, a trainable model wrapped by TrainOneStepCell, whose sprcific type of layer is transformed to the corresponding bayesian layer.
        """
```

参数`dnn_layer`指定将哪个类型的DNN层转换成BNN层，`bnn_layer`指定DNN层将转换成哪个类型的BNN层，`get_args`和`add_args`分别指定从DNN层中获取哪些参数和要为BNN层的哪些参数重新赋值。

在MindSpore中将DNN模型中的Dense层转换成相应贝叶斯层`DenseReparam`的代码如下：

```python
train_bnn_network = bnn_transformer.transform_to_bnn_layer(nn.Dense, bnn_layers.DenseReparam)
```

转换后网络的结构如下：

```text
LeNet5
  (conv1) Conv2dinput_channels=1, output_channels=6, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (conv2) Conv2dinput_channels=6, output_channels=16, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (fc1) DenseReparam
    in_channels=400, out_channels=120, weight_mean=Parameter (name=fc1.weight_posterior.mean), weight_std=Parameter (name=fc1.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc1.bias_posterior.mean), bias_std=Parameter (name=fc1.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc2) DenseReparam
    in_channels=120, out_channels=84, weight_mean=Parameter (name=fc2.weight_posterior.mean), weight_std=Parameter (name=fc2.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc2.bias_posterior.mean), bias_std=Parameter (name=fc2.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc3) DenseReparam
    in_channels=84, out_channels=10, weight_mean=Parameter (name=fc3.weight_posterior.mean), weight_std=Parameter (name=fc3.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc3.bias_posterior.mean), bias_std=Parameter (name=fc3.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
```

可以看到，LeNet网络中的卷积层保持不变，全连接层变成了对应的贝叶斯层`DenseReparam`。

## 使用不确定性估计工具箱

贝叶斯神经网络的优势之一就是可以获取不确定性，MDP在上层提供了不确定性估计的工具箱，用户可以很方便地使用该工具箱计算不确定性。不确定性意味着深度学习模型对预测结果的不确定程度。目前，大多数深度学习算法只能给出预测结果，而不能判断预测结果的可靠性。不确定性主要有两种类型：偶然不确定性和认知不确定性。

- 偶然不确定性（Aleatoric Uncertainty）：描述数据中的内在噪声，即无法避免的误差，这个现象不能通过增加采样数据来削弱。
- 认知不确定性（Epistemic Uncertainty）：模型自身对输入数据的估计可能因为训练不佳、训练数据不够等原因而不准确，可以通过增加训练数据等方式来缓解。

不确定性估计工具箱，适用于主流的深度学习模型，如回归、分类等。在推理阶段，利用不确定性估计工具箱，开发人员只需通过训练模型和训练数据集，指定需要估计的任务和样本，即可得到任意不确定性和认知不确定性。基于不确定性信息，开发人员可以更好地理解模型和数据集。
> 本例面向GPU或Ascend 910 AI处理器平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/toolbox>

以分类任务为例，本例中使用的模型是LeNet，数据集为MNIST，数据处理过程与教程中的[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/quick_start/quick_start.html)一致。为了评估测试示例的不确定性，使用工具箱的方法如下:

```python
from mindspore.nn.probability.toolbox.uncertainty_evaluation import UncertaintyEvaluation
from mindspore import load_checkpoint, load_param_into_net

network = LeNet5()
param_dict = load_checkpoint('checkpoint_lenet.ckpt')
load_param_into_net(network, param_dict)
# get train and eval dataset
ds_train = create_dataset('workspace/mnist/train')
ds_eval = create_dataset('workspace/mnist/test')
evaluation = UncertaintyEvaluation(model=network,
                                   train_dataset=ds_train,
                                   task_type='classification',
                                   num_classes=10,
                                   epochs=1,
                                   epi_uncer_model_path=None,
                                   ale_uncer_model_path=None,
                                   save_model=False)
for eval_data in ds_eval.create_dict_iterator():
    eval_data = Tensor(eval_data['image'].asnumpy(), mstype.float32)
    epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
    aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
```
