# 训练一个LeNet模型

`Linux` `Android` `全流程` `模型导出` `模型转换` `模型训练` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/quick_start/train_lenet.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

本教程基于[LeNet训练示例代码](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/lite/examples/train_lenet)，演示MindSpore Lite训练功能的使用。

整个端侧训练流程分为以下三步：

1. 基于MindSpore构建训练模型，并导出`MindIR`格式文件。
2. 使用MindSpore Lite `Converter`工具，将`MindIR`模型转为端侧`MS`模型。
3. 调用MindSpore Lite训练API，加载端侧`MS`模型，执行训练。

下面章节首先通过示例代码中集成好的脚本，帮你快速部署并执行示例，再详细讲解实现细节。

## 准备

以下操作均在PC上完成，推荐使用x86平台的Ubuntu 18.04 64位操作系统。

### 下载数据集

我们示例中用到的`MNIST`数据集是由10类28*28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。

> MNIST数据集下载页面：<http://yann.lecun.com/exdb/mnist/>。页面提供4个数据集下载链接，其中前2个文件是训练数据需要，后2个文件是测试结果需要。

将数据集下载并解压到本地路径下，这里将数据集解压分别存放到`/PATH/MNIST_Data/train`、`/PATH/MNIST_Data/test`路径下。

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

安装MindSpore CPU环境，具体请参考[MindSpore安装](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_cpu_install_pip.md#)。

### 获取Converter和Runtime

可以通过MindSpore Lite[源码编译](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/build.html)生成模型训练所需的`train-linux-x64`以及`train-android-aarch64`包。编译命令如下：

```shell
# 生成converter工具以及x86平台的runtime包
bash build.sh -I x86_64 -T on -e cpu -j8

# 生成arm64平台的runtime包
bash build.sh -I arm64 -T on -e cpu -j8
```

你也可以[下载MindSpore Lite](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/downloads.html)直接下载所需要的转换工具以及模型训练框架，并将它们放在MindSpore源码下的`output`目录（如果没有`output`目录，请创建它）。

### 连接安卓设备

准备好一台Android设备，并通过USB与工作电脑正确连接。手机需开启“USB调试模式”，华为手机一般在`设置->系统和更新->开发人员选项->USB调试`中打开“USB调试模式”。

本示例使用[`adb`](https://developer.android.google.cn/studio/command-line/adb)工具与Android设备进行通信，在工作电脑上远程执行各类设备操作；如果没有安装`adb`工具，可以执行`apt install adb`安装。

## 模型训练和验证

示例代码在MindSpore[源码](https://gitee.com/mindspore/mindspore)下的`mindspore/lite/examples/train_lenet`目录。本地克隆MindSpore源码后，进入`mindspore/lite/examples/train_lenet`目录，执行如下命令后，脚本会导出`lenet_tod.mindir`模型，然后利用`converter`工具将`MindIR`模型转换为MindSpore Lite可以识别的`lenet_tod.ms`模型；最后，将`lenet_tod.ms`模型文件、MNIST数据集以及MindSpore Lite训练runtime包推送到Andorid设备上，执行训练。

```bash
bash prepare_and_run.sh -D /PATH/MNIST_Data -t arm64
```

其中，`/PATH/MNIST_Data`是你工作电脑上存放MNIST数据集的绝对路径，`-t arm64`表示我们将在Android设备上执行训练和推理。

在Android设备上训练LeNet模型每100轮会输出损失值和准确率；最后选择训练完成的模型执行推理，验证`MNIST`手写字识别精度。在端侧训练的LeNet模型能够达到97%的识别率，结果如下所示（测试准确率会受设备差异的影响）：

```text
Training on Device
100:    Loss is 0.853509 [min=0.581739]  max_acc=0.674079
200:    Loss is 0.729228 [min=0.350235]  max_acc=0.753305
300:    Loss is 0.379949 [min=0.284498]  max_acc=0.847957
400:    Loss is 0.773617 [min=0.186403]  max_acc=0.867788
500:    Loss is 0.477829 [min=0.0688716]  max_acc=0.907051
600:    Loss is 0.333066 [min=0.0688716]  max_acc=0.93099
700:    Loss is 0.197988 [min=0.0549653]  max_acc=0.940905
800:    Loss is 0.128299 [min=0.048147]  max_acc=0.946314
900:    Loss is 0.43212 [min=0.0427626]  max_acc=0.955729
1000:   Loss is 0.446575 [min=0.033213]  max_acc=0.95643
1100:   Loss is 0.162593 [min=0.025461]  max_acc=0.95643
1200:   Loss is 0.177662 [min=0.0180249]  max_acc=0.95643
1300:   Loss is 0.0425688 [min=0.00832943]  max_acc=0.95643
1400:   Loss is 0.270186 [min=0.00832943]  max_acc=0.963041
1500:   Loss is 0.0340949 [min=0.00832943]  max_acc=0.963041
1600:   Loss is 0.205415 [min=0.00832943]  max_acc=0.969551
1700:   Loss is 0.0269625 [min=0.00810314]  max_acc=0.970152
1800:   Loss is 0.197761 [min=0.00680999]  max_acc=0.970152
1900:   Loss is 0.19131 [min=0.00680999]  max_acc=0.970152
2000:   Loss is 0.182704 [min=0.00680999]  max_acc=0.970453
2100:   Loss is 0.375163 [min=0.00313038]  max_acc=0.970453
2200:   Loss is 0.296488 [min=0.00313038]  max_acc=0.970453
2300:   Loss is 0.0556241 [min=0.00313038]  max_acc=0.970453
2400:   Loss is 0.0753383 [min=0.00313038]  max_acc=0.973057
2500:   Loss is 0.0732852 [min=0.00313038]  max_acc=0.973057
2600:   Loss is 0.220644 [min=0.00313038]  max_acc=0.973057
2700:   Loss is 0.0159947 [min=0.00313038]  max_acc=0.973257
2800:   Loss is 0.0800904 [min=0.00168969]  max_acc=0.973257
2900:   Loss is 0.0210299 [min=0.00168969]  max_acc=0.97476
3000:   Loss is 0.256663 [min=0.00168969]  max_acc=0.97476
accuracy = 0.970553

Load trained model and evaluate accuracy
accuracy = 0.970553
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
  ├── scripts
  │   ├── eval.sh
  │   ├── run_eval.sh
  │   ├── train.sh
  │   └── run_train.sh
  │
  ├── src
  │   ├── dataset.cc
  │   ├── dataset.h
  │   ├── net_runner.cc
  │   └── net_runner.h
  │
  ├── README.md
  └── prepare_and_run.sh
```

### 定义并导出模型

首先我们基于MindSpore框架创建一个LeNet5模型，你也可以直接用MindSpore model_zoo的现有[LeNet5模型](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/lenet)。

> 本小节完全使用MindSpore云侧功能，进一步了解MindSpore请参考[MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/index.html)。

```python
import sys
from mindspore import context, Tensor, export
from mindspore import dtype as mstype
from lenet import LeNet5
import numpy as np
from train_utils import TrainWrap

sys.path.append('./mindspore/model_zoo/official/cv/lenet/src/')

n = LeNet5()
n.set_train()
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", save_graphs=False)
```

然后定义输入和标签张量大小：

```python
batch_size = 32
x = Tensor(np.ones((batch_size, 1, 32, 32)), mstype.float32)
label = Tensor(np.zeros([batch_size, 10]).astype(np.float32))
net = TrainWrap(n)
```

定义损失函数、网络可训练参数、优化器，并启用单步训练，由`TrainWrap`函数实现。

```python
import mindspore.nn as nn
from mindspore import ParameterTuple
def TrainWrap(net, loss_fn=None, optimizer=None, weights=None):
    if loss_fn == None:
        loss_fn = nn.SoftMaxCrossEntropyWithLogits()
    loss_net = nn.WithLossCell(net, loss_fn)
    loss_net.set_train()
    if weights == None:
        weights = ParameterTuple(net.trainable_params())
    if optimizer == None:
        optimizer = nn.Adam(weights, learning_rate=1e-3, beta1=0.9 beta2=0.999, eps=1e-8, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
```

最后调用`export`接口将模型导出为`MindIR`文件保存（目前端侧训练仅支持`MindIR`格式）。

```python
export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
print("finished exporting")
```

如果输出`finished exporting`表示导出成功，生成的`lenet_tod.mindir`文件在当前目录下。完整代码参见`lenet_export.py`和`train_utils.py`。

### 转换模型

得到`lenet_tod.mindir`文件后，使用MindSpore Lite `converter`工具将其转还为可用于端侧训练的模型文件，执行指令如下：

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

转换成功后，当前目录下会生成`lenet_tod.ms`模型。

> 详细的`converter`工具使用，可以参考[训练模型转换](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/converter_train.html)。

### 训练模型

源码[`src/net_runner.cc`](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/examples/train_lenet/src/net_runner.cc)使用MindSpore Lite训练API完成模型训练，主函数如下：

```cpp
int NetRunner::Main() {
  InitAndFigureInputs();

  InitDB();

  TrainLoop();

  float acc = CalculateAccuracy();
  std::cout << "accuracy = " << acc << std::endl;

  if (cycles_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained_" + std::to_string(cycles_) + ".ms";
    session_->SaveToFile(trained_fn);
  }
  return 0;
}
```

1. 加载模型

    `InitAndFigureInputs`函数加载转换后的`MS`模型文件，调用`CreateSession`接口创建`TrainSession`实例(下述代码中的`ms_file_`就是转换模型阶段生成的`lenet_tod.ms`模型)。同时根据模型的输入Tensor设置`batch_szie`、`data_size`。

    ```cpp
    void NetRunner::InitAndFigureInputs() {
      mindspore::lite::Context context;
      context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
      context.thread_num_ = 1;

      session_ = mindspore::session::TrainSession::CreateSession(ms_file_, &context);
      assert(nullptr != session_);

      auto inputs = session_->GetInputs();
      assert(inputs.size() > 1);
      this->data_index_  = 0;
      this->label_index_ = 1;
      this->batch_size_ = inputs[data_index_]->shape()[0];
      this->data_size_  = inputs[data_index_]->Size() / batch_size_;  // in bytes
      if (verbose_) {
        std::cout << "data size: " << data_size_ << "\nbatch size: " << batch_size_ << std::endl;
      }
    }
    ```

2. 数据集处理

    `InitDB`函数初始化`MNIST`数据集，调用[DataSet](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/examples/train_lenet/src/dataset.cc)加载训练数据以及相应标签。

    ```cpp
    int NetRunner::InitDB() {
      if (data_size_ != 0) ds_.set_expected_data_size(data_size_);
      int ret = ds_.Init(data_dir_, DS_MNIST_BINARY);
      num_of_classes_ = ds_.num_of_classes();
      if (ds_.test_data().size() == 0) {
        std::cout << "No relevant data was found in " << data_dir_ << std::endl;
        assert(ds_.test_data().size() != 0);
      }

      return ret;
    }
    ```

3. 执行训练

    `TrainSession`和`DataSet`创建完毕后，就可以开始训练了。首先调用`TrainSession`的`Train`方法，将模型设置为训练模式；然后循环调用`RunGraph`函数，通过`DataSet`读取训练数据，执行训练；在一轮训练完成后，可以调用`SaveToFile`方法保存成`CheckPoint`模型，`CheckPoint`模型包含已更新的权重，在应用崩溃或设备出现故障时可以直接加载`CheckPoint`模型，继续开始训练。

    ```cpp
    int NetRunner::TrainLoop() {
        session_->Train();
        float  min_loss = 1000.;
        float max_acc = 0;
        for (int i = 0; i < cycles_; i++) {
            FillInputData(ds_.train_data());
            session_->RunGraph(nullptr, verbose_? after_callback : nullptr);
            float loss = GetLoss();
            if (min_loss > loss) {
                min_loss = loss;
            }
            if (save_checkpoint_ != 0 && (i+1)%save_checkpoint_ == 0) {
                auto cpkt_file = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained_" + std::to_string(i+1) + ".ms";
                session_->SaveToFile(cpkt_file);
            }

            std::cout << i + 1 << ": Loss is " << loss[0] << " [min=" << min_loss << "]";
            if (max_acc > 0) {
                std::cout << "max_acc=" << max_acc;
            }
            std::cout << std::endl;
            if ((i+1)%100 == 0) {
                float acc = CalculateAccuracy(10);
                if (max_acc < acc) {
                    max_acc = acc;
                }
                std::cout << "accuracy (on " << batch_size_ * 10 << " samples) = " << acc;
                std::cout << "max accuracy= " << max_acc << std::endl;
            }
        }
        return 0;
    }
    ```

4. 验证精度

    在每一轮训练结束后，都会调用`CalculateAccuracy`评估模型精度。验证精度是执行推理流程，需要首先调用`TrainSession`的`Eval`方法，将当前的模型设置为推理模式；然后通过读取`/PATH/MNIST_Data/test`测试数据集，调用`RunGraph`执行推理，获取模型输出并比对结果得到识别率。为了下一轮的训练能够继续执行，在获取当前阶段的精度后，调用`Train`方法，将`TrainSession`重新设置为训练模式。

    ```cpp
    float NetRunner::CalculateAccuracy(int max_tests) const {
      float accuracy = 0.0;
      const std::vector<DataLabelTuple> test_set = ds_.test_data();
      int tests = test_set.size() / batch_size_;
      if (max_tests != -1 && tests < max_tests) tests = max_tests;

      session_->Eval();
      for (int i = 0; i < tests; i++) {
        auto labels = FillInputData(test_set, (max_tests == -1));
        session_->RunGraph();
        auto outputsv = SearchOutputsForSize(batch_size_ * num_of_classes_);
        assert(outputsv != nullptr);
        auto scores = reinterpret_cast<float *>(outputsv->MutableData());
        for (int b = 0; b < batch_size_; b++) {
          int max_idx = 0;
          float max_score = scores[num_of_classes_ * b];
          for (int c = 0; c < num_of_classes_; c++) {
            if (scores[num_of_classes_ * b + c] > max_score) {
              max_score = scores[num_of_classes_ * b + c];
              max_idx = c;
            }
          }
          if (labels[b] == max_idx) accuracy += 1.0;
        }
      }
      session_->Train();
      accuracy /= static_cast<float>(batch_size_ * tests);
      return accuracy;
    }
    ```
