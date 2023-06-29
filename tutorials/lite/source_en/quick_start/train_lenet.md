# Training a LeNet Model

`Linux` `Android` `Whole Process` `Model Export` `Model Converting` `Model Training` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/lite/source_en/quick_start/train_lenet.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Here we will demonstrate the code that trains a LeNet model using MindSpore Training-on-Device infrastructure. The code segments that are given below are provided fully in [train_lenet](https://gitee.com/mindspore/mindspore/tree/r1.2/mindspore/lite/examples/train_lenet/).

The completed training procedure is as follows:

1. Constructing your training model based on MindSpore Lite Architecture and Export it into `MindIR` model file.
2. Converting `MindIR` model file to the `MS` ToD model file by using MindSpore Lite `Converter` tool.
3. Loading `MS` model file and executing model training by calling MindSpore Lite training API.

Details will be told after environment deployed and model training by running prepared shell scripts.

## Environment Preparing

Ubuntu 18.04 64-bit operating system on x86 platform is recommended.

### DataSet

The `MNIST` dataset used in this example consists of 10 classes of 28 x 28 pixels grayscale images. It has a training set of 60,000 examples, and a test set of 10,000 examples.

> Download the MNIST dataset at <http://yann.lecun.com/exdb/mnist/>. This page provides four download links of dataset files. The first two links are training dataset and training label, while the last two links are test dataset and test label.

Download and decompress the files to `/PATH/MNIST_Data/train` and `/PATH/MNIST_Data/test` separately.

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

MindSpore can be installed by source code or using `pip`. Refer [MindSpore installation guide](https://gitee.com/mindspore/docs/blob/r1.2/install/mindspore_cpu_install_pip_en.md#) for more details.

### Download and Install MindSpore Lite

Use `git` to clone the source code, the command in `Linux` is as follows:

```shell
git clone https://gitee.com/mindspore/mindspore.git -b r1.2
cd ./mindspore
```

The `mindspore/lite/examples/train_lenet` directory relative to the MindSpore Lite source code contains this demo's source code.

Go to the [MindSpore Lite Download Page](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/downloads.html) to download the mindspore-lite-{version}-linux-x64.tar.gz and mindspore-lite-{version}-android-aarch64.tar.gz. The mindspore-lite-{version}-linux-x64.tar.gz is the MindSpore Lite install package for x86 platform, it contains the converter tool `converter_lite`, this demo uses it to converte `MIDIR` model to `.ms` which is supported by MindSpore Lite; The mindspore-lite-{version}-android-aarch64.tar.gz is the MindSpore Lite install package for Android, it contains training runtime library `libmindspore-lite.so`, this demo uses it to train model. After download these two files, you need rename the mindspore-lite-{version}-linux-x64.tar.gz to mindspore-lite-{version}-train-linux-x64.tar.gz and rename the mindspore-lite-{version}-android-aarch64.tar.gz to mindspore-lite-{version}-train-android-aarch64.tar.gz. Then put the renamed files to the `output` directory relative to MindSpore Lite source code（if there is no `output` directory，you should create it).

Suppose these packags are downloaded in `/Downloads` directory, `Linux` commands for operations above is as follows:

```bash
mkdir output
cp /Downloads/mindspore-lite-{version}-linux-x64.tar.gz output/mindspore-lite-{version}-train-linux-x64.tar.gz
cp /Downloads/mindspore-lite-{version}0-android-aarch64.tar.gz output/mindspore-lite-{version}-train-android-aarch64.tar.gz
```

You can also [compile from source](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.2/use/build.html) to generate the training package for x86 platform mindspore-lite-{version}-train-linux-x64.tar.gz and for Andorid platform mindspore-lite-{version}-train-android-aarch64.tar.gz. These packages will directly generated in `output` directory and you should make sure that in the `output` directory both the two packages exist.

### Connect Android Device

Turning on the 'USB debugging' mode of your Android device and connect it with your PC by using `adb` debugging tool (run`sudo apt install adb` in Ubuntu OS command line).

## Train and Eval

Enter the target directory and run the training bash script. The `Linux` command is as follows:

```bash
cd /mindspore/lite/examples/train_lenet
bash prepare_and_run.sh -D /PATH/MNIST_Data -t arm64
```

`/PATH/MNIST_Data` is the absolute mnist dataset path in your machine, `-t arm64` represents that we will train and run the model on an Android device.

The script `prepare_and_run.sh` has done the following works:

1. Export the `lenet_tod.mindir` model file.
2. Calling the converter tool in the last section and convert the `MINDIR` file to the `ms` file.
3. Push the `lenet.ms` model file, MNIST dataset and the related library files to your `Android` device.
4. Train, save and infer the model.

The model will be trained on your device and print training loss and accuracy value every epoch. The trained model will be saved as 'lenet_tod.ms' file. The 10 epochs training result of lenet is shown below (the classification accuracy varies in devices):

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

### Folder Structure

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

Whether it is an off-the-shelf prepared model, or a custom written model, the model needs to be exported to a `.mindir` file. Here we use the already-implemented [LeNet model](https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/official/cv/lenet).

Import and instantiate a LeNet5 model and set the model to train mode:

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

Set MindSpore context and initialize the data and label tensors. In this case we use a MindSpore that was compiled for CPU. We define a batch size of 32 and initialize the tensors according to MNIST data -- single channel 32x32 images.

The tensors does not need to be loaded with relevant data, but the shape and type must be correct. Note also, that this export code runs on the server, and in this case uses the CPU device. However, the Training on Device will run according to the [context](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/runtime_train_cpp.html#creating-contexts)

```python
BATCH_SIZE = 32
x = Tensor(np.ones((BATCH_SIZE, 1, 32, 32)), mstype.float32)
label = Tensor(np.zeros([BATCH_SIZE]).astype(np.int32))
net = TrainWrap(n)
```

Wrapping the network with a loss layer and an optimizer and `export` it to a `MindIR` file. `TrainWrap` is provided in the example as:

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

Finally, exporting the defined model.

```python
export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
print("finished exporting")
```

### Model Transferring

To convert the model simply use the converter as explained in the [Convert Section](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/converter_train.html#creating-mindspore-tod-models), the command is:

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

The exported file `lenet_tod.ms` is under the folder `./train_lenet/model`.

### Model Training

The model training progress is in [net_runner.cc](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/examples/train_lenet/src/net_runner.cc).

The main code continues as follows:

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
    session_->SaveToFile(trained_fn);
  }
  return 0;
}
```

#### Loading Model

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
  loop_ = mindspore::session::TrainLoop::CreateTrainLoop(session_);

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

#### Dataset Processing

`InitDB` initializes the MNIST dataset and loads it into the memory. MindData has provided the data preprocessing API, the user could refer to the [C++ API Docs](https://www.mindspore.cn/doc/api_cpp/en/r1.2/session.html) for more details.

```cpp
int NetRunner::InitDB() {
  train_ds_ = Mnist(data_dir_ + "/train", "all");

  TypeCast typecast_f("float32");
  Resize resize({h_, w_});
  train_ds_ = train_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast("int32");
  train_ds_ = train_ds_->Map({&typecast}, {"label"});

  train_ds_ = train_ds_->Shuffle(2);
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

#### Execute Training

The `TrainLoop` method is the core of the training procedure. We first display its code then review it.

```cpp
int NetRunner::TrainLoop() {
  struct mindspore::lite::StepLRLambda step_lr_lambda(1, 0.7);
  mindspore::lite::LRScheduler step_lr_sched(mindspore::lite::StepLRLambda, static_cast<void *>(&step_lr_lambda), 1);

  mindspore::lite::LossMonitor lm(100);
  mindspore::lite::ClassificationTrainAccuracyMonitor am(1);
  mindspore::lite::CkptSaver cs(1000, std::string("lenet"));
  Rescaler rescale(255.0);

  loop_->Train(epochs_, train_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale, &lm, &cs, &am, &step_lr_sched});
  return 0;
}
```

#### Execute Evaluating

To eval the model accuracy, the `CalculateAccuracy` method is being called. Within which, the model is switched to `Eval` mode, and the method runs a cycle of test tensors through the trained network to measure the current accuracy rate.

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
  std::cout << "Eval Accuracy is " << acc_metrics_->Eval() << std::endl;

  return 0.0;
}
```

In the given example, the program runs a fixed number of train cycles. The user may easily change the termination condition, e.g., run until a certain accuracy is reached, or run only at night time when device is connected to a power source.

Finally, when trainining is completed, the fully trained model needs to be saved. The `SaveToFile` method is used for this purpose.
