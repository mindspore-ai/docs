# 应用差分隐私机制保护用户隐私

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/protect_user_privacy_with_differential_privacy.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

差分隐私是一种保护用户数据隐私的机制。什么是隐私，隐私指的是单个用户的某些属性，一群用户的某一些属性可以不看做隐私。例如：“抽烟的人有更高的几率会得肺癌”，这个不泄露隐私，但是“张三抽烟，得了肺癌”，这个就泄露了张三的隐私。如果我们知道A医院，今天就诊的100个病人，其中有10个肺癌，并且我们知道了其中99个人的患病信息，就可以推测剩下一个人是否患有肺癌。这种窃取隐私的行为叫做差分攻击。差分隐私是防止差分攻击的方法，通过添加噪声，使得差别只有一条记录的两个数据集，通过模型推理获得相同结果的概率非常接近。也就是说，用了差分隐私后，攻击者知道的100个人的患病信息和99个人的患病信息几乎是一样的，从而无法推测出剩下1个人的患病情况。

### 机器学习中的差分隐私

机器学习算法一般是用大量数据并更新模型参数，学习数据特征。在理想情况下，这些算法学习到一些泛化性较好的模型，例如“吸烟患者更容易得肺癌”，而不是特定的个体特征，例如“张三是个吸烟者，患有肺癌”。然而，机器学习算法并不会区分通用特征还是个体特征。当我们用机器学习来完成某个重要的任务，例如肺癌诊断，发布的机器学习模型，可能在无意中透露训练集中的个体特征，恶意攻击者可能从发布的模型获得关于张三的隐私信息，因此使用差分隐私技术来保护机器学习模型是十分必要的。

**差分隐私定义**[1]为：

$Pr[\mathcal{K}(D)\in S] \le e^{\epsilon} Pr[\mathcal{K}(D') \in S]+\delta$

对于两个差别只有一条记录的数据集$D, D'$，通过随机算法$\mathcal{K}$，输出为结果集合$S$子集的概率满足上面公式，$\epsilon$为差分隐私预算，$\delta$ 为扰动，$\epsilon, \delta$越小，$\mathcal{K}$在$D, D'$上输出的数据分布越接近。

### 差分隐私的度量

差分隐私可以用$\epsilon, \delta$ 度量。

- $\epsilon$：数据集中增加或者减少一条记录，引起的输出概率可以改变的上限。我们通常希望$\epsilon$是一个较小的常数，值越小表示差分隐私条件越严格。
- $\delta$：用于限制模型行为任意改变的概率，通常设置为一个小的常数，推荐设置小于训练数据集大小的倒数。

### MindArmour实现的差分隐私

MindArmour的差分隐私模块Differential-Privacy，实现了差分隐私优化器。目前支持基于高斯机制的差分隐私SGD、Momentum、Adam优化器。其中，高斯噪声机制支持固定标准差的非自适应高斯噪声和随着时间或者迭代步数变化而变化的自适应高斯噪声，使用非自适应高斯噪声的优势在于可以严格控制差分隐私预算$\epsilon$，缺点是在模型训练过程中，每个Step添加的噪声量固定，在训练后期，较大的噪声使得模型收敛困难，甚至导致性能大幅下跌，模型可用性差。自适应噪声很好的解决了这个问题，在模型训练初期，添加的噪声量较大，随着模型逐渐收敛，噪声量逐渐减小，噪声对于模型可用性的影响减小。自适应噪声的缺点是不能严格控制差分隐私预算，在同样的初始值下，自适应差分隐私的$\epsilon$比非自适应的大。同时还提供RDP（R’enyi differential privacy）[2]用于监测差分隐私预算。

这里以LeNet模型，MNIST 数据集为例，说明如何在MindSpore上使用差分隐私优化器训练神经网络模型。

> 由于算子支持的限制，差分隐私训练目前只支持在GPU或者Ascend服务器上面进行，不支持CPU。
本例面向Ascend 910 AI处理器，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/mindarmour/blob/master/examples/privacy/diff_privacy/lenet5_dp.py>

## 实现阶段

### 导入需要的库文件

下列是我们需要的公共模块、MindSpore相关模块和差分隐私特性模块。

```python
import os
from easydict import EasyDict as edict

import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.nn import Accuracy
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

from mindarmour.privacy.diff_privacy import DPModel
from mindarmour.privacy.diff_privacy import NoiseMechanismsFactory
from mindarmour.privacy.diff_privacy import ClipMechanismsFactory
from mindarmour.privacy.diff_privacy import PrivacyMonitorFactory
from mindarmour.utils import LogUtil

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Lenet5_train'
```

### 参数配置

1. 设置运行环境、数据集路径、模型训练参数、checkpoint存储参数、差分隐私参数，`data_path`数据路径替换成你的数据集所在路径。更多配置可以参考<https://gitee.com/mindspore/mindarmour/blob/master/examples/privacy/diff_privacy/lenet5_config.py>。

   ```python
   cfg = edict({
        'num_classes': 10,  # the number of classes of model's output
        'lr': 0.01,  # the learning rate of model's optimizer
        'momentum': 0.9,  # the momentum value of model's optimizer
        'epoch_size': 10,  # training epochs
        'batch_size': 256,  # batch size for training
        'image_height': 32,  # the height of training samples
        'image_width': 32,  # the width of training samples
        'save_checkpoint_steps': 234,  # the interval steps for saving checkpoint file of the model
        'keep_checkpoint_max': 10,  # the maximum number of checkpoint files would be saved
        'device_target': 'Ascend',  # device used
        'data_path': '../../common/dataset/MNIST',  # the path of training and testing dataset
        'dataset_sink_mode': False,  # whether deliver all training data to device one time
        'micro_batches': 32,  # the number of small batches split from an original batch
        'norm_bound': 1.0,  # the clip bound of the gradients of model's training parameters
        'initial_noise_multiplier': 0.05,  # the initial multiplication coefficient of the noise added to training
        # parameters' gradients
        'noise_mechanisms': 'Gaussian',  # the method of adding noise in gradients while training
        'clip_mechanisms': 'Gaussian',  # the method of adaptive clipping gradients while training
        'clip_decay_policy': 'Linear', # Decay policy of adaptive clipping, decay_policy must be in ['Linear', 'Geometric'].
        'clip_learning_rate': 0.001, # Learning rate of update norm clip.
        'target_unclipped_quantile': 0.9, # Target quantile of norm clip.
        'fraction_stddev': 0.01, # The stddev of Gaussian normal which used in empirical_fraction.
        'optimizer': 'Momentum'  # the base optimizer used for Differential privacy training
   })
   ```

2. 配置必要的信息，包括环境信息、执行的模式。

   ```python
   context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
   ```

   详细的接口配置信息，请参见`context.set_context`接口说明。

### 预处理数据集

加载数据集并处理成MindSpore数据格式。

```python
def generate_mnist_dataset(data_path, batch_size=32, repeat_size=1,
                           num_parallel_workers=1, sparse=True):
    """
    create dataset for training or testing
    """
    # define dataset
    ds1 = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    if not sparse:
        one_hot_enco = C.OneHot(10)
        ds1 = ds1.map(operations=one_hot_enco, input_columns="label",
                      num_parallel_workers=num_parallel_workers)
        type_cast_op = C.TypeCast(mstype.float32)
    ds1 = ds1.map(operations=type_cast_op, input_columns="label",
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(operations=resize_op, input_columns="image",
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(operations=rescale_op, input_columns="image",
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(operations=hwc2chw_op, input_columns="image",
                  num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    ds1 = ds1.shuffle(buffer_size=buffer_size)
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1
```

### 建立模型

这里以LeNet模型为例，您也可以建立训练自己的模型。

```python
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.05)


class LeNet5(nn.Cell):
    """
    LeNet network
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

加载LeNet网络，定义损失函数、配置checkpoint、用上述定义的数据加载函数`generate_mnist_dataset`载入数据。

```python
network = LeNet5()
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                             directory='./trained_ckpt_file/',
                             config=config_ck)

# get training dataset
ds_train = generate_mnist_dataset(os.path.join(cfg.data_path, "train"),
                                  cfg.batch_size)
```

### 引入差分隐私

1. 配置差分隐私优化器的参数。

    - 判断`micro_batches`和`batch_size`参数是否符合要求，`batch_size`必须要整除`micro_batches`。
    - 实例化差分隐私工厂类。
    - 设置差分隐私的噪声机制，目前mechanisms支持固定标准差的高斯噪声机制：`Gaussian`和自适应调整标准差的高斯噪声机制：`AdaGaussian`。
    - 设置优化器类型，目前支持`SGD`、`Momentum`和`Adam`。
    - 设置差分隐私预算监测器RDP，用于观测每个step中的差分隐私预算$\epsilon$的变化。

    ```python
    if cfg.micro_batches and cfg.batch_size % cfg.micro_batches != 0:
        raise ValueError(
            "Number of micro_batches should divide evenly batch_size")
    # Create a factory class of DP noise mechanisms, this method is adding noise
    # in gradients while training. Initial_noise_multiplier is suggested to be
    # greater than 1.0, otherwise the privacy budget would be huge, which means
    # that the privacy protection effect is weak. Mechanisms can be 'Gaussian'
    # or 'AdaGaussian', in which noise would be decayed with 'AdaGaussian'
    # mechanism while be constant with 'Gaussian' mechanism.
    noise_mech = NoiseMechanismsFactory().create(cfg.noise_mechanisms,
                                                norm_bound=cfg.norm_bound,
                                                initial_noise_multiplier=cfg.initial_noise_multiplier,
                                                decay_policy=None)
    # Create a factory class of clip mechanisms, this method is to adaptive clip
    # gradients while training, decay_policy support 'Linear' and 'Geometric',
    # learning_rate is the learning rate to update clip_norm,
    # target_unclipped_quantile is the target quantile of norm clip,
    # fraction_stddev is the stddev of Gaussian normal which used in
    # empirical_fraction, the formula is
    # $empirical_fraction + N(0, fraction_stddev)$.
    clip_mech = ClipMechanismsFactory().create(cfg.clip_mechanisms,
                                                decay_policy=cfg.clip_decay_policy,
                                                learning_rate=cfg.clip_learning_rate,
                                                target_unclipped_quantile=cfg.target_unclipped_quantile,
                                                fraction_stddev=cfg.fraction_stddev)
    net_opt = nn.Momentum(params=network.trainable_params(),
                            learning_rate=cfg.lr, momentum=cfg.momentum)
    # Create a monitor for DP training. The function of the monitor is to
    # compute and print the privacy budget(eps and delta) while training.
    rdp_monitor = PrivacyMonitorFactory.create('rdp',
                                                num_samples=60000,
                                                batch_size=cfg.batch_size,
                                                initial_noise_multiplier=cfg.initial_noise_multiplier,
                                                per_print_times=234,
                                                noise_decay_mode=None)
    ```

2. 将LeNet模型包装成差分隐私模型，只需要将网络传入`DPModel`即可。

   ```python
   # Create the DP model for training.
   model = DPModel(micro_batches=cfg.micro_batches,
                   norm_bound=cfg.norm_bound,
                   noise_mech=noise_mech,
                   clip_mech=clip_mech,
                   network=network,
                   loss_fn=net_loss,
                   optimizer=net_opt,
                   metrics={"Accuracy": Accuracy()})
   ```

3. 模型训练与测试。

   ```python
    LOGGER.info(TAG, "============== Starting Training ==============")
    model.train(cfg['epoch_size'], ds_train,
                callbacks=[ckpoint_cb, LossMonitor(), rdp_monitor],
                dataset_sink_mode=cfg.dataset_sink_mode)

    LOGGER.info(TAG, "============== Starting Testing ==============")
    ckpt_file_name = 'trained_ckpt_file/checkpoint_lenet-10_234.ckpt'
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(network, param_dict)
    ds_eval = generate_mnist_dataset(os.path.join(cfg.data_path, 'test'),
                                     batch_size=cfg.batch_size)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    LOGGER.info(TAG, "============== Accuracy: %s  ==============", acc)
   ```

4. 运行命令。

   运行脚本，可在命令行输入命令：

   ```bash
   python lenet_dp.py
   ```

   其中`lenet5_dp.py`替换成你的脚本的名字。

5. 结果展示。

   不加差分隐私的LeNet模型精度稳定在99%，加了Gaussian噪声，自适应Clip的差分隐私LeNet模型收敛，精度稳定在95%左右。

   ```text
   ============== Starting Training ==============
   ...
   ============== Starting Testing ==============
   ...
   ============== Accuracy: 0.9698  ==============
   ```

### 引用

[1] C. Dwork and J. Lei. Differential privacy and robust statistics. In STOC, pages 371–380. ACM, 2009.

[2] Ilya Mironov. Rényi diﬀerential privacy. In IEEE Computer Security Foundations Symposium, 2017.

[3] Abadi, M. e. a., 2016. *Deep learning with differential privacy.* s.l.:Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security.
