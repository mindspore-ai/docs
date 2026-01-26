# 基于C++接口实现端侧训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/train/train_lenet.md)

> MindSpore Lite 已经统一端边云推理API，如您想继续使用MindSpore Lite独立API进行端侧训练，可以参考[此文档](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/quick_start/train_lenet.html)。

## 概述

本教程基于[LeNet训练示例代码](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/train_lenet_cpp)，演示在Android设备上训练一个LeNet。

端侧训练流程如下：

1. 基于MindSpore构建训练模型，并导出`MindIR`模型文件。
2. 使用MindSpore Lite `Converter`工具，将`MindIR`模型转为端侧`MS`模型。
3. 调用MindSpore Lite训练API，加载端侧`MS`模型，执行训练。

下面章节首先通过示例代码中集成好的脚本，帮你快速部署并执行示例，再详细讲解实现细节。

## 准备

推荐使用Ubuntu 18.04 64位操作系统。

### 环境要求

- 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02 LTS

- 软件依赖

    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

    - [CMake](https://cmake.org/download/) >= 3.18.3

    - [Git](https://git-scm.com/downloads) >= 2.28.0

    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
        - 配置环境变量：`export ANDROID_NDK=NDK路径`

### 下载数据集

示例中的`MNIST`数据集由10类28*28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。

> MNIST数据集官网下载地址：<http://yann.lecun.com/exdb/mnist/>，共4个下载链接，分别是训练数据、训练标签、测试数据和测试标签。

下载并解压到本地，解压后的训练和测试集分别存放于`/PATH/MNIST_Data/train`和`/PATH/MNIST_Data/test`路径下。

目录结构如下：

```text
./MNIST_Data/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

### 安装MindSpore

你可以通过`pip`或是源码的方式安装MindSpore，详见[MindSpore官网安装教程](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip.md#)。

### 下载并安装MindSpore Lite

通过`git`克隆源码，进入源码目录，`Linux`指令如下：

```bash
git clone https://atomgit.com/mindspore/mindspore-lite.git -b {version}
cd ./mindspore-lite
```

源码路径下的`mindspore-lite/examples/train_lenet_cpp`目录包含了本示例程序的源码。其中version和下文中[MindSpore Lite下载页面](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)的version保持一致。如果-b 指定master，需要通过[源码编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)的方式获取对应的安装包。

请到[MindSpore Lite下载页面](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)下载mindspore-lite-{version}-linux-x64.tar.gz以及mindspore-lite-{version}-android-aarch64.tar.gz。其中，mindspore-lite-{version}-linux-x64.tar.gz是MindSpore Lite在x86平台的安装包，里面包含模型转换工具converter_lite，本示例用它来将MINDIR模型转换成MindSpore Lite支持的`.ms`格式；mindspore-lite-{version}-android-aarch64.tar.gz是MindSpore Lite在Android平台的安装包，里面包含训练运行时库libmindspore-lite.so，本示例用它所提供的接口在Android上训练模型。最后将文件放到MindSpore Lite源码下的`output`目录（如果没有`output`目录，请创建它）。

假设下载的安装包存放在`/Downloads`目录，上述操作对应的`Linux`指令如下：

```bash
mkdir output
cp /Downloads/mindspore-lite-{version}-linux-x64.tar.gz output/mindspore-lite-{version}-linux-x64.tar.gz
cp /Downloads/mindspore-lite-{version}-android-aarch64.tar.gz output/mindspore-lite-{version}-android-aarch64.tar.gz
```

您也可以通过[源码编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)直接生成端侧训练框架对应的x86平台安装包mindspore-lite-{version}-linux-x64.tar.gz以及Android平台安装包mindspore-lite-{version}-android-aarch64.tar.gz，源码编译的安装包会自动生成在`output`目录下，请确保`output`目录下同时存在这两个安装包。

### 连接安卓设备

准备好一台Android设备，并通过USB与工作电脑正确连接。手机需开启“USB调试模式”，华为手机一般在`设置->系统和更新->开发人员选项->USB调试`中打开“USB调试模式”。

本示例使用[adb](https://developer.android.google.cn/studio/command-line/adb)工具与Android设备进行通信，在工作电脑上远程操控移动设备；如果没有安装`adb`工具，可以执行`apt install adb`安装。

## 模型训练和验证

进入示例代码目录并执行训练脚本，`Linux`指令如下：

```bash
cd mindspore-lite/examples/train_lenet_cpp
bash prepare_and_run.sh -D /PATH/MNIST_Data -t arm64
```

其中`/PATH/MNIST_Data`是你工作电脑上存放MNIST数据集的绝对路径，`-t arm64`为执行训练和推理的设备类型，如果工作电脑连接多台手机设备，可使用`-i devices_id`指定运行设备。

`prepare_and_run.sh`脚本做了以下工作：

1. 导出`lenet_tod.mindir`模型文件；
2. 调用上节的模型转换工具将`lenet_tod.mindir`转换为`lenet_tod.ms`文件；
3. 将`lenet_tod.ms`、MNIST数据集和相关依赖库文件推送至你的`Android`设备；
4. 执行训练、保存并推理模型。

Android设备上训练LeNet模型每轮会输出损失值和准确率；最后选择训练完成的模型执行推理，验证`MNIST`手写字识别精度。端侧训练LeNet模型10个epoch的结果如下所示（测试准确率会受设备差异的影响）：

```text
======Training Locally=========
1.100:  Loss is 1.19449
1.200:  Loss is 0.477986
1.300:  Loss is 0.440362
1.400:  Loss is 0.165605
1.500:  Loss is 0.368853
1.600:  Loss is 0.179764
1.700:  Loss is 0.173386
1.800:  Loss is 0.0767713
1.900:  Loss is 0.493
1.1000: Loss is 0.460352
1.1100: Loss is 0.262044
1.1200: Loss is 0.222022
1.1300: Loss is 0.058006
1.1400: Loss is 0.0794117
1.1500: Loss is 0.0241433
1.1600: Loss is 0.127109
1.1700: Loss is 0.0557566
1.1800: Loss is 0.0698758
Epoch (1):      Loss is 0.384778
Epoch (1):      Training Accuracy is 0.8702
2.100:  Loss is 0.0538642
2.200:  Loss is 0.444504
2.300:  Loss is 0.0806976
2.400:  Loss is 0.0495807
2.500:  Loss is 0.178903
2.600:  Loss is 0.265705
2.700:  Loss is 0.0933796
2.800:  Loss is 0.0880472
2.900:  Loss is 0.0480734
2.1000: Loss is 0.241272
2.1100: Loss is 0.0920451
2.1200: Loss is 0.371406
2.1300: Loss is 0.0365746
2.1400: Loss is 0.0784372
2.1500: Loss is 0.207537
2.1600: Loss is 0.442626
2.1700: Loss is 0.0814725
2.1800: Loss is 0.12081
Epoch (2):      Loss is 0.176118
Epoch (2):      Training Accuracy is 0.94415
......
10.1000:        Loss is 0.0984653
10.1100:        Loss is 0.189702
10.1200:        Loss is 0.0896037
10.1300:        Loss is 0.0138191
10.1400:        Loss is 0.0152357
10.1500:        Loss is 0.12785
10.1600:        Loss is 0.026495
10.1700:        Loss is 0.436495
10.1800:        Loss is 0.157564
Epoch (5):     Loss is 0.102652
Epoch (5):     Training Accuracy is 0.96805
AvgRunTime: 18980.5 ms
Total allocation: 125829120
Accuracy is 0.965244

===Evaluating trained Model=====
Total allocation: 20971520
Accuracy is 0.965244

===Running Inference Model=====
There are 1 input tensors with sizes:
tensor 0: shape is [32 32 32 1]
There are 1 output tensors with sizes:
tensor 0: shape is [32 10]
The predicted classes are:
4, 0, 2, 8, 9, 4, 5, 6, 3, 5, 2, 1, 4, 6, 8, 0, 5, 7, 3, 5, 8, 3, 4, 1, 9, 8, 7, 3, 0, 2, 3, 6,
```

> 如果你没有Android设备，也可以执行`bash prepare_and_run.sh -D /PATH/MNIST_Data -t x86`直接在PC上运行本示例。

## 示例程序详解

### 示例程序结构

```text
train_lenet_cpp/
  ├── model
  │   ├── lenet_export.py
  │   ├── prepare_model.sh
  │   └── train_utils.py
  │
  ├── scripts
  │   ├── batch_of32.dat
  │   ├── eval.sh
  │   ├── infer.sh
  │   └── train.sh
  │
  ├── src
  │   ├── inference.cc
  │   ├── net_runner.cc
  │   ├── net_runner.h
  │   └── utils.h
  │
  ├── Makefile
  ├── README.md
  ├── README_CN.md
  └── prepare_and_run.sh
```

### 定义并导出模型

首先我们需要基于MindSpore框架创建一个LeNet模型。

> 本节使用MindSpore云侧功能导出，更多信息请参考[MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)。

```python
import numpy as np
import mindspore as ms
from lenet import LeNet5
from train_utils import TrainWrap

n = LeNet5()
n.set_train()
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")
```

然后定义输入和标签张量大小：

```python
BATCH_SIZE = 32
x = ms.Tensor(np.ones((BATCH_SIZE, 1, 32, 32)), ms.float32)
label = ms.Tensor(np.zeros([BATCH_SIZE]).astype(np.int32))
net = TrainWrap(n)
```

定义损失函数、网络可训练参数、优化器，并启用单步训练，由`TrainWrap`函数实现。

```python
from mindspore import nn
import mindspore as ms

def train_wrap(net, loss_fn=None, optimizer=None, weights=None):
    """
    train_wrap
    """
    if loss_fn is None:
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)
    loss_net = nn.WithLossCell(net, loss_fn)
    loss_net.set_train()
    if weights is None:
        weights = ms.ParameterTuple(net.trainable_params())
    if optimizer is None:
        optimizer = nn.Adam(weights, learning_rate=0.003, beta1=0.9, beta2=0.999, eps=1e-5, use_locking=False, use_nesterov=False, weight_decay=4e-5, loss_scale=1.0)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    return train_net
```

最后调用`export`接口将模型导出为`MindIR`文件保存（目前端侧训练仅支持`MindIR`格式）。

```python
ms.export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
print("finished exporting")
```

如果输出`finished exporting`表示导出成功，生成的`lenet_tod.mindir`文件在`../train_lenet_cpp/model`目录下。完整代码参见`lenet_export.py`和`train_utils.py`。

### 转换模型

在`prepare_model.sh`中使用MindSpore Lite `converter_lite`工具将`lenet_tod.mindir`转换为`ms`模型文件，执行指令如下：

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

转换成功后，当前目录下会生成`lenet_tod.ms`模型文件。

> 更多用法参见[训练模型转换](https://www.mindspore.cn/lite/docs/zh-CN/master/train/converter_train.html)。

### 训练模型

当前`MindSpore Lite`已移除`MindData`及其相关高阶训练接口，包括`Train`、`Evaluate`，以及部分依赖的回调类（如`AccuracyMetrics`、`CkptSaver`、`TrainAccuracy`、`LossMonitor` 等）。
因此，暂不支持通过高阶接口进行模型训练。后续将补充基于 `RunStep` 接口的训练使用说明。

另外，由于`libmindspore-lite-train`与`libmindspore-lite`之间为弱依赖关系，在使用 C++ 接口`RunStep`进行训练时，如需启用训练能力，需要显式强制链接`libmindspore-lite-train`对应的动态库（.so），可通过链接选项`-Wl,--no-as-needed`实现。