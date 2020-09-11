# 成员推理攻击

<!-- TOC -->

- [成员推理攻击](#成员推理攻击)
    - [概述](#概述)
    - [实现阶段](#实现阶段)
        - [导入需要的库文件](#导入需要的库文件)
        - [加载数据集](#加载数据集)
        - [建立模型](#建立模型)
        - [运用MembershipInference](#运用membershipinference)
    - [参考文献](#参考文献)
        
<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced_use/membership_inference.md" target="_blank"><img src="../_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

成员推理攻击是一种窃取用户数据隐私的方法。隐私指的是单个用户的某些属性，一旦泄露可能会造成人身损害、名誉损害等后果。通常情况下，用户的隐私数据会作保密处理，但我们可以利用非敏感信息来进行推测。例如：”抽烟的人更容易得肺癌“，这个信息不属于隐私信息，但如果知道“张三抽烟”，就可以推断“张三”更容易得肺癌，这就是成员推理。

机器学习/深度学习的成员推理攻击(Membership Inference)，指的是攻击者拥有模型的部分访问权限(黑盒、灰盒或白盒)，能够获取到模型的输出、结构或参数等部分或全部信息，并基于这些信息推断某个样本是否属于模型的训练集。

这里以VGG16模型，CIFAR-100数据集为例，说明如何使用MembershipInference。本教程使用预训练的模型参数进行演示，这里仅给出模型结构、参数设置和数据集预处理方式。

>本例面向Ascend 910处理器，您可以在这里下载完整的样例代码：
>
><https://gitee.com/mindspore/mindarmour/blob/master/example/membership_inference_demo/main.py>

## 实现阶段

### 导入需要的库文件
#### 引入相关包
下面是我们需要的公共模块、MindSpore相关模块和MembershipInference特性模块，以及配置日志标签和日志等级。

```python
import argparse
import sys
import math
import os

import numpy as np

import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
from mindarmour.diff_privacy.evaluation.membership_inference import MembershipInference
from mindarmour.utils import LogUtil

LOGGER = LogUtil.get_instance()
TAG = "MembershipInference_test"
LOGGER.set_level("INFO")
```
### 加载数据集

这里采用的是CIFAR-100数据集，您也可以采用自己的数据集，但要保证传入的数据仅有两项属性"image"和"label"。
```python
# Generate CIFAR-100 data.
def vgg_create_dataset100(data_home, image_size, batch_size, rank_id=0, rank_size=1, repeat_num=1,
                          training=True, num_samples=None, shuffle=True):
    """Data operations."""
    de.config.set_seed(1)
    data_dir = os.path.join(data_home, "train")
    if not training:
        data_dir = os.path.join(data_home, "test")

    if num_samples is not None:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id,
                                      num_samples=num_samples, shuffle=shuffle)
    else:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id)

    input_columns = ["fine_label"]
    output_columns = ["label"]
    data_set = data_set.rename(input_columns=input_columns, output_columns=output_columns)
    data_set = data_set.project(["image", "label"])

    rescale = 1.0 / 255.0
    shift = 0.0

    # Define map operations.
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT.
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR.
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # Apply map operations on images.
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # Apply repeat operations.
    data_set = data_set.repeat(repeat_num)

    # Apply batch operations.
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
```
### 建立模型

这里以VGG16模型为例，您也可以替换为自己的模型。
```python
def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight_shape = (v, in_channels, 3, 3)
            weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32).to_tensor()
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train"):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


base16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def vgg16(num_classes=1000, args=None, phase="train"):
    net = Vgg(base16, num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase)
    return net
```

### 运用MembershipInference
1. 构建VGG16模型并加载参数文件。
   
    这里直接加载预训练完成的VGG16参数配置，您也可以使用如上的网络自行训练。
    
    ```python
    ...
    # load parameter
    parser = argparse.ArgumentParser("main case arg parser.")
    parser.add_argument("--data_path", type=str, required=True, help="Data home path for dataset")
    parser.add_argument("--pre_trained", type=str, required=True, help="Checkpoint path")
    args = parser.parse_args()
    args.batch_norm = True
    args.has_dropout = False
    args.has_bias = False
    args.padding = 0
    args.pad_mode = "same"
    args.weight_decay = 5e-4
    args.loss_scale = 1.0
    
    data_path = "./cifar-100-binary"      # Replace your data path here.
    pre_trained = "./VGG16-100_781.ckpt"  # Replace your pre trained checkpoint file here.
    
    # Load the pretrained model.
    net = vgg16(num_classes=100, args=args)
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9,
                      weight_decay=args.weight_decay, loss_scale=args.loss_scale)
    load_param_into_net(net, load_checkpoint(args.pre_trained))
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    ```
    
2. 加载CIFAR-100数据集，按8:2分割为成员推理攻击模型的训练集和测试集。

    ```python
    # Load and split dataset.
    train_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                          batch_size=64, num_samples=10000, shuffle=False)
    test_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                         batch_size=64, num_samples=10000, shuffle=False, training=False)
    train_train, eval_train = train_dataset.split([0.8, 0.2])
    train_test, eval_test = test_dataset.split([0.8, 0.2])
    msg = "Data loading completed."
    LOGGER.info(TAG, msg)
    ```

3. 配置攻击参数和评估参数
   
    设置用于成员推理评估的方法和参数。目前支持的推理方法有：KNN、LR、MLPClassifier和RandomForest Classifier。
    
    ```python
    config = [
            {
                "method": "lr",
                "params": {
                    "C": np.logspace(-4, 2, 10)
                }
            },
        	{
                "method": "knn",
                "params": {
                    "n_neighbors": [3, 5, 7]
                }
            },
            {
                "method": "mlp",
                "params": {
                    "hidden_layer_sizes": [(64,), (32, 32)],
                    "solver": ["adam"],
                    "alpha": [0.0001, 0.001, 0.01]
                }
            },
            {
                "method": "rf",
                "params": {
                    "n_estimators": [100],
                    "max_features": ["auto", "sqrt"],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        ]
    ```
    
    设置评价指标，目前支持3种评价指标。包括：
    * 准确率：accuracy。
    * 精确率：precision。
    * 召回率：recall。
      
    ```python
    metrics = ["precision", "accuracy", "recall"]
    ```
    
4. 训练成员推理攻击模型，并给出评估结果。

    ```python
    attacker = MembershipInference(model)                  # Get attack model.
    
    attacker.train(train_train, train_test, config)        # Train attack model.
    msg = "Membership inference model training completed."
    LOGGER.info(TAG, msg)
    
    result = attacker.eval(eval_train, eval_test, metrics) # Eval metrics.
    count = len(config)
    for i in range(count):
        print("Method: {}, {}".format(config[i]["method"], result[i]))
    ```

5. 实验结果。

    成员推理的指标如下所示，各数值均保留至小数点后四位。

    以第一行结果为例：在使用lr（逻辑回归分类）进行成员推理时，推理的准确率（accuracy）为0.7132，推理精确率（precision）为0.6596，正类样本召回率为0.8810。在二分类任务下，指标表明我们的成员推理是有效的。
    
    ```
    Method: lr, {'recall': 0.8810,'precision': 0.6596,'accuracy': 0.7132}
    Method: knn, {'recall': 0.7082,'precision': 0.5613,'accuracy': 0.5774}
    Method: mlp, {'recall': 0.6729,'precision': 0.6462,'accuracy': 0.6522}
    Method: rf, {'recall': 0.8513, 'precision': 0.6655, 'accuracy': 0.7117}
    ```

## 参考文献
[1] [Shokri R , Stronati M , Song C , et al. Membership Inference Attacks against Machine Learning Models[J].](https://arxiv.org/abs/1610.05820v2)
