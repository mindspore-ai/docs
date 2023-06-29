# 基于C++接口实现端侧训练

`Linux` `C++` `Android` `全流程` `模型导出` `模型转换` `模型训练` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_zh_cn/quick_start/train_lenet.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## 概述

本教程基于[LeNet训练示例代码](https://gitee.com/mindspore/mindspore/tree/r1.3/mindspore/lite/examples/train_lenet)，演示在Android设备上训练一个LeNet。

端侧训练流程如下：

1. 基于MindSpore构建训练模型，并导出`MindIR`模型文件。
2. 使用MindSpore Lite `Converter`工具，将`MindIR`模型转为端侧`MS`模型。
3. 调用MindSpore Lite训练API，加载端侧`MS`模型，执行训练。

下面章节首先通过示例代码中集成好的脚本，帮你快速部署并执行示例，再详细讲解实现细节。

## 准备

推荐使用Ubuntu 18.04 64位操作系统。

### 下载数据集

示例中的`MNIST`数据集由10类28*28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。

> MNIST数据集官网下载地址：<http://yann.lecun.com/exdb/mnist/>，共4个下载链接，分别是训练数据、训练标签、测试数据和测试标签。

下载并解压到本地，解压后的训练和测试集分别存放于`/PATH/MNIST_Data/train`和`/PATH/MNIST_Data/test`路径下。

目录结构如下：

```text
MNIST_Data/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

### 安装MindSpore

你可以通过`pip`或是源码的方式安装MindSpore，详见[MindSpore官网安装教程](https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_cpu_install_pip.md#)。

### 下载并安装MindSpore Lite

通过`git`克隆源码，进入源码目录，`Linux`指令如下：

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
cd ./mindspore
```

源码路径下的`mindspore/lite/examples/train_lenet`目录包含了本示例程序的源码。

请到[MindSpore Lite下载页面](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/downloads.html)下载mindspore-lite-{version}-linux-x64.tar.gz以及mindspore-lite-{version}-android-aarch64.tar.gz。其中，mindspore-lite-{version}-linux-x64.tar.gz是MindSpore Lite在x86平台的安装包，里面包含模型转换工具converter_lite，本示例用它来将MINDIR模型转换成MindSpore Lite支持的`.ms`格式；mindspore-lite-{version}-android-aarch64.tar.gz是MindSpore Lite在Android平台的安装包，里面包含训练运行时库libmindspore-lite.so，本示例用它所提供的接口在Android上训练模型。最后将文件放到MindSpore源码下的`output`目录（如果没有`output`目录，请创建它）。

假设下载的安装包存放在`/Downloads`目录，上述操作对应的`Linux`指令如下：

```bash
mkdir output
cp /Downloads/mindspore-lite-{version}-linux-x64.tar.gz output/mindspore-lite-{version}-linux-x64.tar.gz
cp /Downloads/mindspore-lite-{version}-android-aarch64.tar.gz output/mindspore-lite-{version}-android-aarch64.tar.gz
```

您也可以通过[源码编译](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/build.html)直接生成端侧训练框架对应的x86平台安装包mindspore-lite-{version}-linux-x64.tar.gz以及Android平台安装包mindspore-lite-{version}-android-aarch64.tar.gz，源码编译的安装包会自动生成在`output`目录下，请确保`output`目录下同时存在这两个安装包。

### 连接安卓设备

准备好一台Android设备，并通过USB与工作电脑正确连接。手机需开启“USB调试模式”，华为手机一般在`设置->系统和更新->开发人员选项->USB调试`中打开“USB调试模式”。

本示例使用[`adb`](https://developer.android.google.cn/studio/command-line/adb)工具与Android设备进行通信，在工作电脑上远程操控移动设备；如果没有安装`adb`工具，可以执行`apt install adb`安装。

## 模型训练和验证

进入示例代码目录并执行训练脚本，`Linux`指令如下：

```bash
cd mindspore/lite/examples/train_lenet
bash prepare_and_run.sh -D /PATH/MNIST_Data -t arm64
```

其中`/PATH/MNIST_Data`是你工作电脑上存放MNIST数据集的绝对路径，`-t arm64`为执行训练和推理的设备类型。

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
Epoch (10):     Loss is 0.102652
Epoch (10):     Training Accuracy is 0.96805
Eval Accuracy is 0.965244
===Evaluating trained Model=====
Eval Accuracy is 0.965244
```

> 如果你没有Android设备，也可以执行`bash prepare_and_run.sh -D /PATH/MNIST_Data -t x86`直接在PC上运行本示例。

## 示例程序详解

### 示例程序结构

```text
  train_lenet/
  ├── model
  │   ├── lenet_export.py
  │   ├── prepare_model.sh
  │   └── train_utils.py
  |
  ├── scripts
  │   ├── eval.sh
  │   └── train.sh
  │
  ├── src
  │   ├── net_runner.cc
  │   ├── net_runner.h
  │   └── utils.h
  │
  ├── README.md
  ├── README_CN.md
  └── prepare_and_run.sh
```

### 定义并导出模型

首先我们需要基于MindSpore框架创建一个LeNet模型，本例中直接用MindSpore model_zoo的现有[LeNet模型](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/official/cv/lenet)。

> 本小结使用MindSpore云侧功能导出，更多信息请参考[MindSpore教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/index.html)。

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import export
from lenet import LeNet5
from train_utils import TrainWrap

n = LeNet5()
n.set_train()
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", save_graphs=False)
```

然后定义输入和标签张量大小：

```python
BATCH_SIZE = 32
x = Tensor(np.ones((BATCH_SIZE, 1, 32, 32)), mstype.float32)
label = Tensor(np.zeros([BATCH_SIZE]).astype(np.int32))
net = TrainWrap(n)
```

定义损失函数、网络可训练参数、优化器，并启用单步训练，由`TrainWrap`函数实现。

```python
import mindspore.nn as nn
from mindspore.common.parameter import ParameterTuple

def TrainWrap(net, loss_fn=None, optimizer=None, weights=None):
    """
    TrainWrap
    """
    if loss_fn is None:
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)
    loss_net = nn.WithLossCell(net, loss_fn)
    loss_net.set_train()
    if weights is None:
        weights = ParameterTuple(net.trainable_params())
    if optimizer is None:
        optimizer = nn.Adam(weights, learning_rate=0.003, beta1=0.9, beta2=0.999, eps=1e-5, use_locking=False, use_nesterov=False, weight_decay=4e-5, loss_scale=1.0)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    return train_net
```

最后调用`export`接口将模型导出为`MindIR`文件保存（目前端侧训练仅支持`MindIR`格式）。

```python
export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
print("finished exporting")
```

如果输出`finished exporting`表示导出成功，生成的`lenet_tod.mindir`文件在`../train_lenet/model`目录下。完整代码参见`lenet_export.py`和`train_utils.py`。

### 转换模型

在`prepare_model.sh`中使用MindSpore Lite `converter_lite`工具将`lenet_tod.mindir`转换为`ms`模型文件，执行指令如下：

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

转换成功后，当前目录下会生成`lenet_tod.ms`模型文件。

> 更多用法参见[训练模型转换](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/converter_train.html)。

### 训练模型

模型训练的处理详细流程请参考[net_runner.cc源码](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/examples/train_lenet/src/net_runner.cc)。

模型训练的主函数为：

```cpp
int NetRunner::Main() {
  // Load model and create session
  InitAndFigureInputs();
  // initialize the dataset
  InitDB();
  // Execute the training
  TrainLoop();
  // Evaluate the trained model
  CalculateAccuracy();

  if (epochs_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained.ms";
    // Save the trained model to file
    session_->Export(trained_fn);
  }
  return 0;
}
```

1. 加载模型

    `InitAndFigureInputs`函数加载转换后的`MS`模型文件，调用`CreateTrainSession`接口创建`TrainSession`实例(下述代码中的`ms_file_`就是转换模型阶段生成的`lenet_tod.ms`模型)。

    ```cpp
    void NetRunner::InitAndFigureInputs() {
      mindspore::lite::Context context;
      context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
      context.device_list_[0].device_info_.cpu_device_info_.enable_float16_ = false;
      context.device_list_[0].device_type_ = mindspore::lite::DT_CPU;
      context.thread_num_ = 2;

      session_ = mindspore::session::TrainSession::CreateTrainSession(ms_file_, &context, true);
      MS_ASSERT(nullptr != session_);

      session_->SetupVirtualBatch(virtual_batch_);
      loop_ = mindspore::session::TrainLoop::CreateTrainLoop(session_);

      if (verbose_) {
        loop_->SetKernelCallBack(nullptr, after_callback);
      }
      acc_metrics_ = std::shared_ptr<AccuracyMetrics>(new AccuracyMetrics);

      loop_->Init({acc_metrics_.get()});

      auto inputs = session_->GetInputs();
      MS_ASSERT(inputs.size() > 1);
      auto nhwc_input_dims = inputs.at(0)->shape();
      MS_ASSERT(nhwc_input_dims.size() == 4);
      batch_size_ = nhwc_input_dims.at(0);
      h_ = nhwc_input_dims.at(1);
      w_ = nhwc_input_dims.at(2);
    }
    ```

2. 数据集处理

    `InitDB`函数预处理`MNIST`数据集并加载至内存。MindData提供了数据预处理API，用户可参见[C++ API 说明文档](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/session.html) 获取更多详细信息。

    ```cpp
    int NetRunner::InitDB() {
      train_ds_ = Mnist(data_dir_ + "/train", "all");

      TypeCast typecast_f("float32");
      Resize resize({h_, w_});
      train_ds_ = train_ds_->Map({&resize, &typecast_f}, {"image"});

      TypeCast typecast("int32");
      train_ds_ = train_ds_->Map({&typecast}, {"label"});

      train_ds_ = train_ds_->Batch(batch_size_, true);

      if (verbose_) {
        std::cout << "DatasetSize is " << train_ds_->GetDatasetSize() << std::endl;
      }
      if (train_ds_->GetDatasetSize() == 0) {
        std::cout << "No relevant data was found in " << data_dir_ << std::endl;
        MS_ASSERT(train_ds_->GetDatasetSize() != 0);
      }
      return 0;
    }
    ```

3. 执行训练

    首先创建训练回调类对象（例如`LRScheduler`、`LossMonitor`、`ClassificationTrainAccuracyMonitor`和`CkptSaver`）数组指针；然后调用`TrainLoop`类的`Train`函数，将模型设置为训练模式；最后在训练过程中遍历执行回调类对象对应的函数并输出训练日志。`CkptSaver`会根据设定训练步长数值为当前会话保存`CheckPoint`模型，`CheckPoint`模型包含已更新的权重，在应用崩溃或设备出现故障时可以直接加载`CheckPoint`模型，继续开始训练。

    ```cpp
    int NetRunner::TrainLoop() {
      mindspore::lite::LossMonitor lm(100);
      mindspore::lite::ClassificationTrainAccuracyMonitor am(1);
      mindspore::lite::CkptSaver cs(1000, std::string("lenet"));
      Rescaler rescale(255.0);

      Measurement measure(epochs_);

      if (virtual_batch_ > 0) {
        loop_->Train(epochs_, train_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale, &lm, &cs, &am, &measure});
      } else {
        struct mindspore::lite::StepLRLambda step_lr_lambda(1, kGammaFactor);
        mindspore::lite::LRScheduler step_lr_sched(mindspore::lite::StepLRLambda, static_cast<void *>(&step_lr_lambda), 1);
        loop_->Train(epochs_, train_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale, &lm, &cs, &am, &step_lr_sched, &measure});
      }
      return 0;
    }
    ```

4. 验证精度

    训练结束后调用`CalculateAccuracy`评估模型精度。该函数调用`TrainSession`的`Eval`方法，将模型设置为推理模式。

    ```cpp
    float NetRunner::CalculateAccuracy(int max_tests) {
      test_ds_ = Mnist(data_dir_ + "/test", "all");
      TypeCast typecast_f("float32");
      Resize resize({h_, w_});
      test_ds_ = test_ds_->Map({&resize, &typecast_f}, {"image"});

      TypeCast typecast("int32");
      test_ds_ = test_ds_->Map({&typecast}, {"label"});
      test_ds_ = test_ds_->Batch(batch_size_, true);

      Rescaler rescale(255.0);

      loop_->Eval(test_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale});
      std::cout << "Accuracy is " << acc_metrics_->Eval() << std::endl;

      return 0.0;
    }
    ```
