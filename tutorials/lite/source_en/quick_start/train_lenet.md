# Training a LeNet Model

`Linux` `Android` `Whole Process` `Model Export` `Model Converting` `Model Training` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_en/quick_start/train_lenet.md)

## Overview

Here we will demonstrate the code that trains a LeNet model using MindSpore Training-on-Device infrastructure. The code segments that are given below are provided fully in [MindSpore gitee](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/lite/examples/train_lenet/).

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

Please referring MindSpore [installation](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_cpu_install_pip_en.md#) to install MindSpore CPU environment.

### Converter and Runtime Tool

Acquire `train-converter-linux-x64` and `train-android-aarch64` tool-package based on MindSpore Lite architecture, refer to [source building](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/build.html) chapter, the command is shown below:

```shell
# generate converter tools and runtime package on x86
bash build.sh -I x86_64 -T on -e cpu -j8

# generate runtime package on arm64
bash build.sh -I arm64 -T on -e cpu -j8
```

You could also directly [download MindSpore Lite](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/downloads.html) and store them in the `output` directory related to the MindSpore source code (if no `output` directory exists, please create it).

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

> If the Android device is not available on your hand, you could also exectute `bash prepare_and_run.sh -D /PATH/MNIST_Data -t x86` and run it on the x86 platform.

## Details

The demo project folder structure:

```bash
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

### Model Exporting

Whether it is an off-the-shelf prepared model, or a custom written model, the model needs to be exported to a `.mindir` file. Here we use the already-implemented [LeNet model](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/lenet).

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

The tensors does not need to be loaded with relevant data, but the shape and type must be correct. Note also, that this export code runs on the server, and in this case uses the CPU device. However, the Training on Device will run according to the [context](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/runtime_train_cpp.html#creating-contexts)

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

To run this python code one must have an installed [MindSpore environment](https://gitee.com/mindspore/mindspore/blob/r1.1/README.md#installation). In the example below we use a CPU-supported MindSpore environment installed on a docker with image name `${DOCKER_IMG}`. Please refer to [MindSpore Docker Image Installation instructions](https://gitee.com/mindspore/mindspore/blob/r1.1/README.md#docker-image).

> MindSpore environment allows the developer to run MindSpore python code on server or PC. It differs from MindSpore Lite framework that allows to compile and run code on embedded devices.

```bash
DOCKER_IMG=$1
echo "============Exporting=========="
docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "python transfer_learning_export.py; chmod 444 transfer_learning_tod.mindir"
```

If you don't have docker environment, it will run locally.

To convert the model simply use the converter as explained in the [Convert Section](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/converter_train.html#creating-mindspore-tod-models)

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_tod.mindir --outputFile=lenet_tod
```

### Model Training

In the [example c++ code](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/lite/examples/train_lenet/src) the executable has the following API:

```bash
Usage: net_runner -f <.ms model file> -d <data_dir> [-c <num of training cycles>]
                 [-v (verbose mode)] [-s <save checkpoint every X iterations>]
```

After parsing the input parameters the main code continues as follows:

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

#### Load Model

`InitAndFigureInputs` creates the TrainSession instance from the `.ms` file, then sets the input tensors indices for the `.ms` model.

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

#### Dataset Processing

`InitDB` initializes the MNIST dataset and loads it into the memory. We will not discuss this code here.
The user may refer to the [code in gitee](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/examples/train_lenet/src/dataset.cc). In the next release, MindData framework will be integrated into this example.

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

#### Execute Training

The `TrainLoop` method is the core of the training procedure. We first display its code then review it.

```cpp
int NetRunner::TrainLoop() {
  session_->Train();
  float min_loss = 1000.;
  float max_acc = 0.;
  for (int i = 0; i < cycles_; i++) {
    FillInputData(ds_.train_data());
    session_->RunGraph(nullptr, verbose_? after_callback : nullptr);
    float loss = GetLoss();
    if (min_loss > loss) min_loss = loss;

    if (save_checkpoint_ != 0 && (i + 1) % save_checkpoint_ == 0) {
      auto cpkt_file = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained_" + std::to_string(i + 1) + ".ms";
      session_->SaveToFile(cpkt_file);
    }

    if ((i + 1) % 100 == 0) {
      float acc = CalculateAccuracy(10);
      if (max_acc < acc) max_acc = acc;
      std::cout << i + 1 << ":\tLoss is " << std::setw(7) << loss << " [min=" << min_loss << "] " << " max_acc=" << max_acc << std::endl;
    }
  }
  return 0;
}
```

Within this section of code the session is switched to train mode using the `Train()` method, then the main loop over all the training cycles takes place. In each cycle, the data is read from the training dataset and loaded into the input tensors. Both data and label are filled in.

```cpp
FillInputData(ds_.train_data());
```

Then, `RunGraph` method is called. A debug callback that prints the input and output tensors is provided if program is launched in verbose mode.

```cpp
session_->RunGraph(nullptr, verbose_? after_callback : nullptr);
```

Following the train cycle, the loss is [extracted from the Output Tensors](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/runtime_train_cpp.html#obtaining-output-tensors).
It is advised to periodically save intermediate training results, i.e., checkpoint files. These files might be handy if the application or device crashes during the training process. The checkpoint files are practically `.ms` files that contain the updated weights, and the program may be relaunched with the checkpoint file as the `.ms` model file. Checkpoints are easily saved by calling the `SaveToFile` API, like this:

```cpp
session_->SaveToFile(cpkt_file);
```

To keep track of the model accuracy, the `CalculateAccuracy` method is being called. Within which, the model is switched to `Eval` mode, and the method runs a cycle of test tensors through the trained network to measure the current accuracy rate. Since this method is time consuming it is not advised to run it every train cycle.

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

In the given example, the program runs a fixed number of train cycles. The user may easily change the termination condition, e.g., run until a certain accuracy is reached, or run only at night time when device is connected to a power source.

Finally, when trainining is completed, the fully trained model needs to be saved. The `SaveToFile` method is used for this purpose.
