# C++ Interface Sample

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/train/train_lenet.md)

> MindSpore has unified the end-to-side cloud inference API. If you want to continue to use the MindSpore Lite independent API for training, you can refer to [here](https://www.mindspore.cn/lite/docs/en/r1.3/quick_start/train_lenet.html).

## Overview

This tutorial is based on [LeNet training example code](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/train_lenet_cpp) and demonstrates training a LeNet on an Android device .

The completed training procedure is as follows:

1. Constructing your training model based on MindSpore Lite Architecture and Export it into `MindIR` model file.
2. Converting `MindIR` model file to the `MS` ToD model file by using MindSpore Lite `Converter` tool.
3. Loading `MS` model file and executing model training by calling MindSpore Lite training API.

Details will be told after environment deployed and model training by running prepared shell scripts.

## Environment Preparing

Ubuntu 18.04 64-bit operating system on x86 platform is recommended.

### Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02LTS is recommended.

- Software dependency

    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

    - [CMake](https://cmake.org/download/) >= 3.18.3

    - [Git](https://git-scm.com/downloads) >= 2.28.0

    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
        - Configure environment variables: `export ANDROID_NDK=NDK path`.

### Downloading the Dataset

The `MNIST` dataset used in this example consists of 10 classes of 28 x 28 pixels grayscale images. It has a training set of 60,000 examples, and a test set of 10,000 examples.

> Download the MNIST dataset at <http://yann.lecun.com/exdb/mnist/>. This page provides four download links of dataset files. The first two links are training dataset and training label, while the last two links are test dataset and test label.

Download and decompress the files to `/PATH/MNIST_Data/train` and `/PATH/MNIST_Data/test` separately.

The directory structure is as follows:

```text
./MNIST_Data/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

### Installing MindSpore

MindSpore can be installed by source code or using `pip`. Refer to [MindSpore installation guide](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_install_pip_en.md#) for more details.

### Downloading and Installing MindSpore Lite

Use `git` to clone the source code, the command in `Linux` is as follows:

```bash
git clone https://gitee.com/mindspore/mindspore.git -b {version}
cd ./mindspore
```

The `mindspore/lite/examples/train_lenet_cpp` directory relative to the MindSpore Lite source code contains this demo's source code. The version is consistent with that of [MindSpore Lite Download Page](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) below. If -b the master is specified, you need to obtain the corresponding installation package through [compile from source](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html).

Go to the [MindSpore Lite Download Page](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) to download the mindspore-lite-{version}-linux-x64.tar.gz and mindspore-lite-{version}-android-aarch64.tar.gz. The mindspore-lite-{version}-linux-x64.tar.gz is the MindSpore Lite install package for x86 platform, it contains the converter tool `converter_lite`, this demo uses it to converte `MIDIR` model to `.ms` which is supported by MindSpore Lite; The mindspore-lite-{version}-android-aarch64.tar.gz is the MindSpore Lite install package for Android, it contains training runtime library `libmindspore-lite.so`, this demo uses it to train model. Then put the files to the `output` directory relative to MindSpore Lite source code (if there is no `output` directory, you should create it).

Suppose these packags are downloaded in `/Downloads` directory, `Linux` commands for operations above is as follows:

```bash
mkdir output
cp /Downloads/mindspore-lite-{version}-linux-x64.tar.gz output/mindspore-lite-{version}-linux-x64.tar.gz
cp /Downloads/mindspore-lite-{version}-android-aarch64.tar.gz output/mindspore-lite-{version}-android-aarch64.tar.gz
```

You can also [compile from source](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html) to generate the training package for x86 platform mindspore-lite-{version}-linux-x64.tar.gz and for Andorid platform mindspore-lite-{version}-android-aarch64.tar.gz. These packages will directly generated in `output` directory and you should make sure that in the `output` directory both the two packages exist.

### Connecting Android Device

Prepare an Android device and connect it properly to the working computer via USB. The phone needs to turn on "USB debugging mode", and Huawei phone usually turns on "USB debugging mode" in `Settings->System and Updates->Developer Options->USB debugging`.

This example uses the [adb](https://developer.android.google.cn/studio/command-line/adb) tool to communicate with an Android device to remotely control the mobile device from a work computer. If you don't have the `adb` tool installed, you can run `apt install adb`.

## Model Training and Evaluation

Enter the target directory and run the training bash script. The `Linux` command is as follows:

```bash
cd mindspore/lite/examples/train_lenet_cpp
bash prepare_and_run.sh -D /PATH/MNIST_Data -t arm64
```

`/PATH/MNIST_Data` is the absolute mnist dataset path in your machine, `-t arm64` represents that we will train and run the model on an Android device, if the work computer is connected to multiple mobile devices, you can use `-i devices_id` to specify the running device.

The script `prepare_and_run.sh` has done the following works:

1. Export the `lenet_tod.mindir` model file.
2. Calling the converter tool in the last section and convert the `MINDIR` file to the `ms` file.
3. Push the `lenet.ms` model file, MNIST dataset and the related library files to your `Android` device.
4. Train, save and infer the model.

The model will be trained on your device and print training loss and accuracy value every epoch. The trained model will be saved as 'lenet_tod.ms' file. The 10 epochs training result of lenet is shown below (the classification accuracy varies in devices):

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

> If the Android device is not available on your hand, you could also exectute `bash prepare_and_run.sh -D /PATH/MNIST_Data -t x86` and run it on the x86 platform.

## Demo Project Details

### Demo Project Folder Structure

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

### Model Exporting

Whether it is an off-the-shelf prepared model, or a custom written model, the model needs to be exported to a `.mindir` file. Here we use the already-implemented [LeNet model](https://gitee.com/mindspore/models/tree/master/research/cv/lenet).

> This summary is exported using the MindSpore cloud side feature. For more information, please refer to [MindSpore Tutorial](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/index.html).

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

Then define the input and label tensor sizes:

```python
BATCH_SIZE = 32
x = ms.Tensor(np.ones((BATCH_SIZE, 1, 32, 32)), ms.float32)
label = ms.Tensor(np.zeros([BATCH_SIZE]).astype(np.int32))
net = TrainWrap(n)
```

Define the loss function, network trainable parameters, optimizer, and enable single-step training, implemented by the `TrainWrap` function.

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

Wrapping the network with a loss layer and an optimizer and `export` it to a `MindIR` file. `TrainWrap` is provided in the example as:

```python
ms.export(net, x, label, file_name="lenet_tod", file_format='MINDIR')
print("finished exporting")
```

If the output `finished exporting` indicates that the export was successful, the generated `lenet_tod.mindir` file is in the `... /train_lenet_cpp/model` directory. See `lenet_export.py` and `train_utils.py` for the complete code.

### Model Transferring

Convert `lenet_tod.mindir` to `ms` model file using MindSpore Lite `converter_lite` tool in `prepare_model.sh` by executing the command as follows:

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

After successful conversion, the `lenet_tod.ms` model file is generated in the current directory.

> See [training model conversion](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/converter_train.html) for more usage.

### Model Training

The model training progress is in [net_runner.cc](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/train_lenet_cpp/src/net_runner.cc).

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
    mindspore::Serialization::ExportModel(*model_, mindspore::kMindIR, trained_fn, mindspore::kNoQuant, false);
    trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_infer.ms";
    mindspore::Serialization::ExportModel(*model_, mindspore::kMindIR, trained_fn, mindspore::kNoQuant, true);
  }
  return 0;
}
```

1. Loading Model

    `InitAndFigureInputs` creates the TrainSession instance from the `.ms` file, then sets the input tensors indices for the `.ms` model.

    ```cpp
    void NetRunner::InitAndFigureInputs() {
      auto context = std::make_shared<mindspore::Context>();
      auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
      cpu_context->SetEnableFP16(enable_fp16_);
      context->MutableDeviceInfo().push_back(cpu_context);

      graph_ = new mindspore::Graph();
      auto status = mindspore::Serialization::Load(ms_file_, mindspore::kMindIR, graph_);
      if (status != mindspore::kSuccess) {
        std::cout << "Error " << status << " during serialization of graph " << ms_file_;
        MS_ASSERT(status != mindspore::kSuccess);
      }

      auto cfg = std::make_shared<mindspore::TrainCfg>();
      if (enable_fp16_) {
        cfg.get()->optimization_level_ = mindspore::kO2;
      }

      model_ = new mindspore::Model();
      status = model_->Build(mindspore::GraphCell(*graph_), context, cfg);
      if (status != mindspore::kSuccess) {
        std::cout << "Error " << status << " during build of model " << ms_file_;
        MS_ASSERT(status != mindspore::kSuccess);
      }

      acc_metrics_ = std::shared_ptr<AccuracyMetrics>(new AccuracyMetrics);
      model_->InitMetrics({acc_metrics_.get()});

      auto inputs = model_->GetInputs();
      MS_ASSERT(inputs.size() >= 1);
      auto nhwc_input_dims = inputs.at(0).Shape();

      batch_size_ = nhwc_input_dims.at(0);
      h_ = nhwc_input_dims.at(1);
      w_ = nhwc_input_dims.at(2);
    }
    ```

2. Dataset Processing

    `InitDB` initializes the MNIST dataset and loads it into the memory. MindData has provided the data preprocessing API, the user could refer to the [C++ API Docs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_dataset.html) for more details.

    ```cpp
    int NetRunner::InitDB() {
      train_ds_ = Mnist(data_dir_ + "/train", "all", std::make_shared<SequentialSampler>(0, 0));

      TypeCast typecast_f(mindspore::DataType::kNumberTypeFloat32);
      Resize resize({h_, w_});
      train_ds_ = train_ds_->Map({&resize, &typecast_f}, {"image"});

      TypeCast typecast(mindspore::DataType::kNumberTypeInt32);
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

3. Execute Training

    First create pointers to an array of training callback class objects (e.g., `LRScheduler`, `LossMonitor`, `TrainAccuracy`, and `CkptSaver`); then call the `Train` function of the `TrainLoop` class to set the model into training mode; and finally iterate through the execution of functions corresponding to the callback class objects during training and outputs the training log. `CkptSaver` saves the `CheckPoint` model for the current session according to the set training step value. The `CheckPoint` model contains the updated weights, so that the `CheckPoint` model can be loaded directly when the application crashes or the device malfunctions, and training can continue.

    ```cpp
    int NetRunner::TrainLoop() {
      mindspore::LossMonitor lm(100);
      mindspore::TrainAccuracy am(1);

      mindspore::CkptSaver cs(kSaveEpochs, std::string("lenet"));
      Rescaler rescale(kScalePoint);
      Measurement measure(epochs_);

      if (virtual_batch_ > 0) {
        model_->Train(epochs_, train_ds_, {&rescale, &lm, &cs, &measure});
      } else {
        struct mindspore::StepLRLambda step_lr_lambda(1, kGammaFactor);
        mindspore::LRScheduler step_lr_sched(mindspore::StepLRLambda, static_cast<void *>(&step_lr_lambda), 1);
        model_->Train(epochs_, train_ds_, {&rescale, &lm, &cs, &am, &step_lr_sched, &measure});
      }

      return 0;
    }
    ```

4. Execute Evaluating

    To eval the model accuracy, the `CalculateAccuracy` method is being called. Within which, the model is switched to `Eval` mode, and the method runs a cycle of test tensors through the trained network to measure the current accuracy rate.

    ```cpp
    float NetRunner::CalculateAccuracy(int max_tests) {
      test_ds_ = Mnist(data_dir_ + "/test", "all");
      TypeCast typecast_f(mindspore::DataType::kNumberTypeFloat32);
      Resize resize({h_, w_});
      test_ds_ = test_ds_->Map({&resize, &typecast_f}, {"image"});

      TypeCast typecast(mindspore::DataType::kNumberTypeInt32);
      test_ds_ = test_ds_->Map({&typecast}, {"label"});
      test_ds_ = test_ds_->Batch(batch_size_, true);

      model_->Evaluate(test_ds_, {});
      std::cout << "Accuracy is " << acc_metrics_->Eval() << std::endl;

      return 0.0;
    }
    ```
