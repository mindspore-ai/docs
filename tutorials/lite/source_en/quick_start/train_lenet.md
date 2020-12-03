# Training a LeNet Model

<!-- TOC -->

- [Overview](#overview)
- [Exporting the model to a .mindir file](#exporting-the-model-to-a-mindir-file)
- [Converting the .mindir file into a ToD loadable .ms file](#converting-the-mindir-file-into-a-tod-loadable-ms-file)
- [Dataset](#dataset)
- [Implementing the main code and train loop](#implementing-the-main-code-and-train-loop)
- [Preparing and running](#preparing-and-running)
    - [Preparing the model](#preparing-the-model)
    - [Preparing the folder](#preparing-the-folder)
    - [Compiling the code](#compiling-the-code)
    - [Running the code on the device](#running-the-code-on-the-device)
    - [Output](#output)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/quick_start/train_lenet.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Here we will explain the code that trains a LeNet model using Training-on-Device infrastructure.
The code segements that are given below are provided fully in [MindSpore gitee](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/train_lenet/).

## Exporting the model to a .mindir file

Whether it is an off-the-shelf prepared model, or a custom written model, the model needs to be exported to a `.mindir` file. Here we use the already-implemented [LeNet model](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet).

Import and instantiate a LeNet5 model, and set the model to train mode:

```python
import sys
sys.path.append('../../../cv/lenet/src/')
from lenet import LeNet5
n = LeNet5()
n.set_train()
```

Set MindSpore context and initialize the data and label tensors. In this case we use a MindSpore that was compiled for GPU. We define a batch size of 32 and initialize the tesnors according to MNIST data -- single channel 32x32 images.

The tensors does not need to be loaded with relevant data, but the shape and type must be correct. Note also, that this export code runs on the server, and in this case uses the GPU device. However, the Training on Device will run according to the [context](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_cpp.html#creating-contexts) provided in the training program.

```python
import mindspore as M
import numpy as np
M.context.set_context(mode=M.context.PYNATIVE_MODE,device_target="GPU", save_graphs=False)
batch_size = 32
x=M.Tensor(np.ones((batch_size,1,32,32)), M.float32)
label = M.Tensor(np.zeros([batch_size, 10]).astype(np.float32))
```

Finally, wrap the network with a Loss layer and an optimizer and `export` it to a `MINDIR` file.

```python
from mindspore.train.serialization import export
from train_utils import TrainWrap
net = TrainWrap(n)
export(net, x, label, file_name="lenet_tod.mindir", file_format='MINDIR')
```

where `TrainWrap` is provided in the example as:

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

To run this python code one must have an installed [MindSpore environment](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation). In the example below we use a GPU-supported MindSpore environment indtalled on a docker with image name `${DOCKER_IMG}`. Please refer to [MindSpore Docker Image Instalation instructions](https://gitee.com/mindspore/mindspore/blob/master/README.md#docker-image).
> MindSpore environment allows the developer to run MindSpore python code on server or PC. It differs from MindSpore Lite framework that allows to compile and run code on embedded devices.

```bash
DOCKER_IMG=$1
echo "============Exporting=========="
docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "python transfer_learning_export.py; chmod 444 transfer_learning_tod.mindir"
```

## Converting the .mindir file into a ToD(Train on Device) loadable .ms file

To convert the model simply use the converter as explained in the [Convert Section](https://www.mindspore.cn/tutorial/lite/en/master/use/convert_model.html)

```bash
CONVERTER="../../../../../mindspore/lite/build/tools/converter/converter_lite"
if [ ! -f "$CONVERTER" ]; then
    echo "$CONVERTER does not exist."
fi

echo "============Converting========="
$CONVERTER --fmk=MINDIR --trainModel=true --modelFile=transfer_learning_tod.mindir --outputFile=transfer_learning_tod
```

## Dataset

In this example we use the MNIST dataset of handwritten digits as published in [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/).

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Train：60,000 images
    - Test：10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.cc

- The dataset directory structure is as follows:

```python
mnist/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

## Implementing the main code and train loop

In the [example c++ code](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/train_lenet/src) the executable has the following API:

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

where `InitAndFigureInputs` creates the TrainSession instance from the `.ms` file, then sets the input tensors indices for the `.ms` model, in our case, LeNet network as well as several derived variables

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

`InitDB` initializes the MNIST dataset and loads it into the memory. We will not discuss this code here.
The user may refer to the [code in gitee](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/train_lenet/src/dataset.cc). In the next release, MindData framework will be integrated into this example.

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

    if (save_checkpoint_ != 0 && (i+1)%save_checkpoint_ == 0) {
      auto cpkt_file = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained_" + std::to_string(i+1) + ".ms";
      session_->SaveToFile(cpkt_file);
    }

    if ((i+1)%100 == 0) {
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

Then, `RunGraph` method is called. A debug callback that prints the input and output tensors is provided if program is launched in verbose mode

```cpp
  session_->RunGraph(nullptr, verbose_? after_callback : nullptr);
```

Following the train cycle, the loss is [extracted from the Output Tensors](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime_cpp.html#obtaining-output-tensors).
It is advised to periodically save intermediate training results, i.e., checkpoint files. These files might be handy if the application or device crashes during the training process. The checkpoint files are practically `.ms` files that contain the updated weights, and the program may be relaunched with the checkpoint file as the `.ms` model file. Checkpoints are easily saved by calling the `SaveToFile` API, like this:

```cpp
  session_->SaveToFile(cpkt_file);
```

To keep track of the model accuracy, the `CalculateAccuracy` method is being called. Within which, the model is switched to `Eval` mode, and the method runs a cycle of test tensors through the trained network to measure the current accuracy rate. Since this method is time consuming it is not advised to run it every train cycle

In the given example, the program runs a fixed number of train cycles. The user may easily change the termination condition, e.g., run until a certain accuracy is reached, or run only at night time when device is connected to a power source.

Finally, when trainining is completed, the fully trained model needs to be saved. The `SaveToFile` method is used for this purpose.

## Preparing and running

The code example provided in the [train_lenet directory](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/train_lenet/) includes a `prepare_and_run.sh` script that performs the followings:

- Prepare a folder that should be sent to the device
- Push this folder into the device
- Run the scripts on the device

The script accepts three paramaters:

- MNIST data directory.
- MindSpore docker image.
- A relaease tar file. Use the [downloaded](https://www.mindspore.cn/tutorial/lite/en/master/use/downloads.html) tar file or the [compiled one](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#output-description)

### Preparing the model

Within the [model directory](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/train_lenet/model) a `prepare_model.sh` script runs the python code that exports the model using a pre-prepared docker and then converts the model using the converter.

```bash
# Prepare the model
cd model/
rm -f *.ms
./prepare_model.sh $DOCKER
cd -
```

### Preparing the Folder

The `lenet_tod.ms` model file is then copied into the `package` folder as well as scripts, the MindSpore ToD library and the MNIST dataset.

```bash
# Copy the .ms model to the package folder
rm -rf package
mkdir -p package/model
cp model/*.ms package/model

# Copy the running script to the package
cp scripts/train.sh package/
cp scripts/eval.sh package/

# Copy the shared MindSpore ToD library
tar -xzvf ${TARBALL} --wildcards --no-anchored libmindspore-lite.so
mv mindspore-*/lib package/
rm -rf mindspore-*

# Copy the dataset to the package
cp -r ${MNIST_DATA_PATH} package/dataset
```

### Compiling the code

Finally, the code is compiled for arm64 and the binary is copied into the `package` folder.

```bash
# Compile program
make TARGET=arm64

# Copy the executable to the package
mv bin package/
```

### Running the code on the device

To run the code on the device simply use adb to push `package` folder into the device and run training and evaluation subsequently

```bash
# Push the folder to the device
adb push package /data/local/tmp/

echo "Training on Device"
adb shell < scripts/run_train.sh

echo
echo "Load trained model and evaluate accuracy"
adb shell < scripts/run_eval.sh
echo
```

### Output

The output on the device should look like this

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
