# Training a LeNet Model

`Linux` `Android` `Whole Process` `Model Export` `Model Converting` `Model Training` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Overview](#overview)
- [Environment Preparing](#environment-preparing)
    - [Dataset](#dataset)
    - [Install MindSpore](#install-mindspore)
    - [Converter and Runtime Tool](#converter-and-runtime-tool)
    - [Connect Android Device](#connect-android-device)
- [Train and Eval](#train-and-eval)
- [Details](#details)
    - [Model Exporting](#model-exporting)
    - [Model Transferring](#model-transferring)
    - [Model Training](#model-training)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/quick_start/train_lenet.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Here we will demonstrate the code that trains a LeNet model using MindSpore Training-on-Device infrastructure. The code segments that are given below are provided fully in [MindSpore gitee](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/train_lenet/).

The completed training procedure is as follows:

1. Constructing your training model based on MindSpore Lite Architecture and Export it into `MindIR` model file.
2. Converting `MindIR` model file to the `MS` ToD model file by using MindSpore Lite `Converter` tool.
3. Loading `MS` model file and executing model training by calling MindSpore Lite training API.

Details will be told after environment deployed and model training by running prepared shell scripts.

## Environment Preparing

All the following operations are under PC, the Ubuntu 18.04 64-bit operating system on x86 platform is recommended.

### DataSet

The `MNIST` dataset used in this example consists of 10 classes of 28 x 28 pixels grayscale images. It has a training set of 60,000 examples, and a test set of 10,000 examples.

> Download the MNIST dataset at <http://yann.lecun.com/exdb/mnist/>. This page provides four download links of dataset files. The first two links are for data training, and the last two links are for data test.

Download the files, decompress them, and store them in the workspace directories `/PATH/MNIST_Data/train` and `/PATH/MNIST_Data/test`.

The directory structure is as follows:

```text
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
            train-images.idx3-ubyte
            train-labels.idx1-ubyte
```

### Install MindSpore

Please referring MindSpore [installation](https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip_en.md#) to install MindSpore CPU environment.

### Converter and Runtime Tool

Acquire `train-converter-linux-x64` and `train-android-aarch64` tool-package based on MindSpore Lite architecture, refer to [source building](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html) chapter, the command is shown below:

```shell
# generate converter tools and runtime package on x86
bash build.sh -I x86_64 -T on -e cpu -j8

# generate runtime package on arm64
bash build.sh -I arm64 -T on -e cpu -j8
```

You could also directly [download MindSpore Lite](https://www.mindspore.cn/tutorial/lite/en/master/use/downloads.html) and store them in the `output` directory related to the MindSpore source code (if no `output` directory exists, please create it).

### Connect Android Device

Turning on the 'USB debugging' mode of your Android device and connect it with your PC by using `adb` debugging tool (run`sudo apt install adb` in Ubuntu OS command line).

## Train and Eval

Executing the bash command below under `./mindspore/lite/example/train_lenet` directory.

```bash
bash prepare_and_run.sh -D /PATH/MNIST_Data -t arm64
```

`/PATH/MNIST_Data` is the absolute mnist dataset path in your machine, `-t arm64` represents that we will train and run the model on an Android device.

The model will be trained on your device and print training loss and accuracy value every 100 epochs. The trained model will be saved as 'lenet_tod.ms' file. The classification accuracy varies in devices.

```bash
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

> If the Android device is not available on your hand, you could also exectute `bash prepare_and_run.sh -D /PATH/MNIST_Data -t x86` and run it on the x86 platform.

## Details

The demo project folder structure:

```bash
train_lenet/
  ├── model
  │   ├── lenet_export.py
  │   ├── prepare_model.sh
  │   └── train_utils.py
  │
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

### Model Exporting

Whether it is an off-the-shelf prepared model, or a custom written model, the model needs to be exported to a `.mindir` file. Here we use the already-implemented [LeNet model](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet).

Import and instantiate a LeNet5 model and set the model to train mode:

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

Set MindSpore context and initialize the data and label tensors. In this case we use a MindSpore that was compiled for CPU. We define a batch size of 32 and initialize the tensors according to MNIST data -- single channel 32x32 images.

The tensors does not need to be loaded with relevant data, but the shape and type must be correct. Note also, that this export code runs on the server, and in this case uses the CPU device. However, the Training on Device will run according to the [context](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_train_cpp.html#creating-contexts)

```python
batch_size = 32
x = Tensor(np.ones((batch_size, 1, 32, 32)), mstype.float32)
label = Tensor(np.zeros([batch_size, 10]).astype(np.float32))
net = TrainWrap(n)
```

Wrapping the network with a loss layer and an optimizer and `export` it to a `MindIR` file. `TrainWrap` is provided in the example as:

```python
import mindspore.nn as nn
from mindspore.common.parameter import ParameterTuple

def TrainWrap(net, loss_fn=None, optimizer=None, weights=None):
  if loss_fn == None:
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
  loss_net = nn.WithLossCell(net, loss_fn)
  loss_net.set_train()
  if weights == None:
    weights = ParameterTuple(net.trainable_params())
  if optimizer == None:
     optimizer = nn.Adam(weights, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
  train_net = nn.TrainOneStepCell(loss_net, optimizer)
```

Finally, exporting the defined model.

```python
export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
print("finished exporting")
```

### Model Transferring

To run this python code one must have an installed [MindSpore environment](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation). In the example below we use a CPU-supported MindSpore environment installed on a docker with image name `${DOCKER_IMG}`. Please refer to [MindSpore Docker Image Installation instructions](https://gitee.com/mindspore/mindspore/blob/master/README.md#docker-image).

> MindSpore environment allows the developer to run MindSpore python code on server or PC. It differs from MindSpore Lite framework that allows to compile and run code on embedded devices.

```bash
DOCKER_IMG=$1
echo "============Exporting=========="
docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "python transfer_learning_export.py; chmod 444 transfer_learning_tod.mindir"
```

If you don't have docker environment, it will run locally.

To convert the model simply use the converter as explained in the [Convert Section](https://www.mindspore.cn/tutorial/lite/en/master/use/converter_train.html#creating-mindspore-tod-models)

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

### Model Training

In the [example c++ code](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/train_lenet/src) the executable has the following API:

```bash
Usage: net_runner -f <.ms model file> -d <data_dir> [-e <num of training epochs>]
                 [-v (verbose mode)] [-s <save checkpoint every X iterations>]
```

After parsing the input parameters the main code continues as follows:

```cpp
int NetRunner::Main() {
  InitAndFigureInputs();

  InitDB();

  TrainLoop();

  CalculateAccuracy()yuchuli;

  if (epochs_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained.ms";
    session_->SaveToFile(trained_fn);
  }
  return 0;
}
```

#### Load Model

`InitAndFigureInputs` creates the TrainSession instance from the `.ms` file, then sets the input tensors indices for the `.ms` model.

```cpp
void NetRunner::InitAndFigureInputs() {
  mindspore::lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
  context.device_list_[0].device_info_.cpu_device_info_.enable_float16_ = false;
  context.device_list_[0].device_type_ = mindspore::lite::DT_CPU;
  context.thread_num_ = 2;

  session_ = mindspore::session::TrainSession::CreateSession(ms_file_, &context);
  MS_ASSERT(nullptr != session_);
  loop_ = mindspore::session::TrainLoop::CreateTrainLoop(session_, &context);

  acc_metrics_ = std::shared_ptr<AccuracyMetrics>(new AccuracyMetrics);

  loop_->Init({acc_metrics_.get()});

  auto inputs = session_->GetInputs();
  MS_ASSERT(inputs.size() > 1);
}
```

#### Dataset Processing

`InitDB` initializes the MNIST dataset and loads it into the memory. MindData has provided the data preprocessing API, the user could refer to the [C++ API Docs](https://www.mindspore.cn/doc/api_cpp/en/master/session.html) for more details.

```cpp
int NetRunner::InitDB() {
  train_ds_ = Mnist(data_dir_ + "/train", "all");
  TypeCast typecast_f("float32");
  Resize resize({32, 32});
  train_ds_ = train_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast("int32");
  train_ds_ = train_ds_->Map({&typecast}, {"label"});

  train_ds_ = train_ds_->Shuffle(2);
  train_ds_ = train_ds_->Batch(32, true);

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

#### Execute Training

The `TrainLoop` method is the core of the training procedure. We first display its code then review it.

```cpp
int NetRunner::TrainLoop() {
  struct mindspore::lite::StepLRLambda step_lr_lambda(1, 0.9);
  mindspore::lite::LRScheduler step_lr_sched(mindspore::lite::StepLRLambda, static_cast<void *>(&step_lr_lambda), 100);

  mindspore::lite::LossMonitor lm(100);
  mindspore::lite::ClassificationTrainAccuracyMonitor am(1);
  mindspore::lite::CkptSaver cs(1000, std::string("lenet"));
  Rescaler rescale(255.0);

  loop_->Train(epochs_, train_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale, &lm, &cs, &am, &step_lr_sched});
  return 0;
}
```

Within this section of code the session is switched to train mode using the `Train()` method, then the main loop over all the training cycles takes place. In each cycle, the data and label are read from the training dataset and loaded into the input tensors. Following the train cycle, the loss is [extracted from the Output Tensors](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_train_cpp.html#obtaining-output-tensors).
It is advised to periodically save intermediate training results, i.e., checkpoint files. These files might be handy if the application or device crashes during the training process. The checkpoint files are practically `.ms` files that contain the updated weights, and the program may be relaunched with the checkpoint file as the `.ms` model file. The user could also save checkpoints manually by calling the `SaveToFile` API, like this:

```cpp
session_->SaveToFile(cpkt_file);
```

To eval the model accuracy, the `CalculateAccuracy` method is being called. Within which, the model is switched to `Eval` mode, and the method runs a cycle of test tensors through the trained network to measure the current accuracy rate.

```cpp
float NetRunner::CalculateAccuracy(int max_tests) {
  test_ds_ = Mnist(data_dir_ + "/test", "all");
  TypeCast typecast_f("float32");
  Resize resize({32, 32});
  test_ds_ = test_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast("int32");
  test_ds_ = test_ds_->Map({&typecast}, {"label"});
  test_ds_ = test_ds_->Batch(32, true);

  Rescaler rescale(255.0);

  loop_->Eval(test_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale});
  std::cout << "Eval Accuracy is " << acc_metrics_->Eval() << std::endl;

  return 0.0;
}
```

In the given example, the program runs a fixed number of train cycles. The user may easily change the termination condition, e.g., run until a certain accuracy is reached, or run only at night time when device is connected to a power source.

Finally, when trainining is completed, the fully trained model needs to be saved. The `SaveToFile` method is used for this purpose.
