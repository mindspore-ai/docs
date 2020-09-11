# 深度概率编程
`Ascend` `GPU` `全流程` `初级` `中级` `高级`

<!-- TOC -->

- [深度概率编程](#深度概率编程)
    - [概述](#概述)
    - [贝叶斯神经网络](#贝叶斯神经网络)
        - [处理数据集](#处理数据集)
        - [定义损失函数和优化器](#定义损失函数和优化器)
        - [训练网络](#训练网络)
    - [变分推断](#变分推断)
        - [定义变分自编码器](#定义变分自编码器)
        - [定义损失函数和优化器](#定义损失函数和优化器)
        - [处理数据](#处理数据)
        - [训练网络](#训练网络)
    - [DNN一键转换成BNN](#DNN一键转换成BNN)
        - [定义DNN模型](#定义DNN模型)
        - [定义损失函数和优化器](#定义损失函数和优化器)
        - [功能一：转换整个模型](#功能一：转换整个模型)
        - [功能二：转换指定类型的层](#功能二：转换指定类型的层)
    - [不确定性估计](#不确定性估计)

<!-- /TOC -->

## 概述
MindSpore深度概率编程（MindSpore Deep Probabilistic Programming, MDP）的目标是将深度学习和贝叶斯学习结合，并能面向不同的开发者。具体来说，对于专业的贝叶斯学习用户，提供概率采样、推理算法和模型构建库；另一方面，为不熟悉贝叶斯深度学习的用户提供了高级的API，从而不用更改深度学习编程逻辑，即可利用贝叶斯模型。

本章将详细介绍深度概率编程在MindSpore上的应用。

### 贝叶斯神经网络
本例子利用贝叶斯神经网络实现一个简单的图片分类功能，整体流程如下：
1. 处理MNIST数据集。
2. 定义贝叶斯LeNet网络。
3. 定义损失函数和优化器。
4. 加载数据集并进行训练。

#### 处理数据集
本例子使用的是MNIST数据集，数据处理过程与教程中的[实现一个图片分类应用](https://www.mindspore.cn/tutorial/zh-CN/master/quick_start/quick_start.html)一致。

#### 定义贝叶斯神经网络
本例子使用的是贝叶斯LeNet。利用bnn_layers构建贝叶斯神经网络的方法与构建普通的神经网络相同。值得注意的是，bnn_layers和普通的神经网络层可以互相组合。

```
import mindspore.nn as nn
from mindspore.nn.probability import bnn_layers
import mindspore.ops.operations as P

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
        self.reshape = P.Reshape()

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
#### 定义损失函数和优化器
接下来需要定义损失函数（Loss）和优化器（Optimizer）。损失函数是深度学习的训练目标，也叫目标函数，可以理解为神经网络的输出（Logits）和标签(Labels)之间的距离，是一个标量数据。
常见的损失函数包括均方误差、L2损失、Hinge损失、交叉熵等等。图像分类应用通常采用交叉熵损失（CrossEntropy）。
优化器用于神经网络求解（训练）。由于神经网络参数规模庞大，无法直接求解，因而深度学习中采用随机梯度下降算法（SGD）及其改进算法进行求解。MindSpore封装了常见的优化器，如SGD、Adam、Momemtum等等。本例采用Adam优化器，通常需要设定两个参数，学习率（learnin _rate）和权重衰减项（weight decay）。
MindSpore中定义损失函数和优化器的代码样例如下：

```
# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
```

#### 训练网络
贝叶斯神经网络的训练过程与DNN基本相同，唯一不同的是将`WithLossCell`替换为适用于BNN的`WithBNNLossCell`。除了`backbone`和`loss_fn`两个参数之外，`WithBNNLossCell`增加了`dnn_factor`和`bnn_factor`两个参数。`dnn_factor`是由损失函数计算得到的网络整体损失的系数，`bnn_factor`是每个贝叶斯层的KL散度的系数，这两个参数是用来平衡网络整体损失和贝叶斯层的KL散度的，防止KL散度的值过大掩盖了网络整体损失。

```
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

```
def train_model(train_net, net, dataset):
    accs = []
    loss_sum = 0
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        loss = train_net(train_x, label)
        output = net(train_x)
        log_output = P.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)
        loss_sum += loss.asnumpy()

    loss_sum = loss_sum / len(accs)
    acc_mean = np.mean(accs)
    return loss_sum, acc_mean


def validate_model(net, dataset):
    accs = []
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        output = net(train_x)
        log_output = P.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)

    acc_mean = np.mean(accs)
    return acc_mean
```

### 变分推断
#### 定义变分自编码器
我们只需要自定义编码器和解码器，编码器和解码器都是神经网络。

```
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
        self.reshape = P.Reshape()

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
接下来需要定义损失函数（Loss）和优化器（Optimizer）。本例使用的损失函数是ELBO，ELBO是变分推断专用的损失函数；本例使用的优化器是Adam。
MindSpore中定义损失函数和优化器的代码样例如下：

```
# loss function definition
net_loss = ELBO(latent_prior='Normal', output_prior='Normal')

# optimization definition
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)

net_with_loss = nn.WithLossCell(vae, net_loss)
```
#### 处理数据
本例使用的是MNIST数据集，数据处理过程与教程中的[实现一个图片分类应用](https://www.mindspore.cn/tutorial/zh-CN/master/quick_start/quick_start.html)一致。

#### 训练网络
使用`SVI`接口对VAE网络进行训练。

```
from mindspore.nn.probability.infer import SVI

vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
vae = vi.run(train_dataset=ds_train, epochs=10)
trained_loss = vi.get_train_loss()
```
通过`vi.run`可以得到训练好的网络，使用`vi.get_train_loss`可以得到训练之后的损失。
#### 生成新样本或重构输入样本
利用训练好的VAE网络，我们可以生成新的样本或重构输入样本。

```
IMAGE_SHAPE = (-1, 1, 32, 32)
generated_sample = vae.generate_sample(64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    reconstructed_sample = vae.reconstruct_sample(sample_x)
```

### DNN一键转换成BNN
对于不熟悉贝叶斯模型的DNN研究人员，MDP提供了高级API`TransformToBNN`，支持DNN模型一键转换成BNN模型。
#### 定义DNN模型
本例使用的DNN模型是LeNet。

```
from mindspore.common.initializer import TruncatedNormal
import mindspore.nn as nn
import mindspore.ops.operations as P

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
        self.reshape = P.Reshape()

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
#### 定义损失函数和优化器
接下来需要定义损失函数（Loss）和优化器（Optimizer）。本例使用交叉熵损失作为损失函数，Adam作为优化器。

```
network = LeNet5()

# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)

net_with_loss = WithLossCell(network, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)
```
#### 实例化TransformToBNN
`TransformToBNN`的`__init__`函数定义如下：

```
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

```
from mindspore.nn.probability import transforms

bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
```
#### 功能一：转换整个模型
`transform_to_bnn_model`方法可以将整个DNN模型转换为BNN模型。其定义如下:

```
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

```
train_bnn_network = bnn_transformer.transform_to_bnn_model()
```
#### 功能二：转换指定类型的层
`transform_to_bnn_layer`方法可以将DNN模型中指定类型的层（nn.Dense或者nn.Conv2d）转换为对应的贝叶斯层。其定义如下:

```
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

### 不确定性估计
不确定性估计工具箱基于MindSpore Deep probability Programming (MDP)，适用于主流的深度学习模型，如回归、分类、目标检测等。在推理阶段，利用不确定性估计工具箱，开发人员只需通过训练模型和训练数据集，指定需要估计的任务和样本，即可得到任意不确定性（aleatoric uncertainty）和认知不确定性（epistemic uncertainty）。基于不确定性信息，开发人员可以更好地理解模型和数据集。
以分类任务为例，本例中使用的模型是LeNet，数据集为MNist，数据处理过程与教程中的[实现一个图片分类应用](https://www.mindspore.cn/tutorial/zh-CN/master/quick_start/quick_start.html)一致。为了评估测试示例的不确定性，使用工具箱的方法如下:

```
from mindspore.nn.probability.toolbox.uncertainty_evaluation import UncertaintyEvaluation
from mindspore.train.serialization import load_checkpoint, load_param_into_net

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
    eval_data = Tensor(eval_data['image'], mstype.float32)
    epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
    aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
```


