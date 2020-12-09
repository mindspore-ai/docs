# Transfer Learning

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

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training_on_device/source_en/code_example/transfer_learning.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Here we will explain the code that performs transfer learning using Training-on-Device infrastructure.
The code segements that are given below are provided fully in [MindSpore gitee](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/transfer_learning/).
The model that is used as the backbone for the transfer learning is [efficientNet](https://arxiv.org/abs/1905.11946).

## Exporting the model to a .mindir file

In this code example we are focusing on Transfer learning, i.e., using an already trained model, in which some of the layers (backbone) will be kept fixed while other layers (head) will be changing throughtout the training process. This distinction between backbone layers and head layers is user defined, and depends on the initial model and transfer learning task. Code-wise it should be defined during the export of the model into a `.mindir` file. As mentioned above, we use an [efficientNet](https://arxiv.org/abs/1905.11946) which is locally defined (within model directory). The pretrained weights which will be loaded into the transfer learning model are available [here](https://download.mindspore.cn/model_zoo/official/lite/efficient_net/efficient_net_b0.ckpt). They are downloaded using wget in the first time the script is run.
The export code goes as follows:

The transfer network is defined as such that have two elements, a backbone and a head:

```python
import mindspore as M

class TransferNet(M.nn.Cell):
  def __init__(self, backbone, head):
    super(TransferNet, self).__init__()
    self.backbone = backbone
    self.head = head
  def construct(self, x):
    x = self.backbone(x)
    x = self.head(x)
    return x
```

The backbone is created and loaded with pretrained data. The head is initialized with random data:

```python
from effnet import effnet
from mindspore.train.serialization import load_checkpoint
import numpy as np
backbone = effnet(num_classes=1000)
load_checkpoint("efficient_net_b0.ckpt", backbone)
head = M.nn.Dense(1000, 10)
head.weight.set_data(M.Tensor(np.random.normal(0, 0.1, head.weight.data.shape).astype("float32")))
head.bias.set_data(M.Tensor(np.zeros(head.bias.data.shape, dtype="float32")))
```

The network is then created with both elements and the trainable parameters are chosen, namely the head:

```python
n = TransferNet(backbone, head)

from mindspore.common.parameter import ParameterTuple
trainable_weights_list = []
trainable_weights_list.extend(n.head.trainable_params())
trainable_weights = ParameterTuple(trainable_weights_list)
```

Mindspore contest is set, in this case we use a MindSpore that was compiled for GPU.
We define a batch size of 32 and initialize the data and label tesnors according to Efficient Net data size -- three channels 224x224 images.
The tensors does not need to be loaded with relevant data, but the shape and type must be correct.

```python
M.context.set_context(mode=M.context.PYNATIVE_MODE,device_target="GPU", save_graphs=False)
BATCH_SIZE = 32
x=M.Tensor(np.ones((BATCH_SIZE,3,224,224)), M.float32)
label = M.Tensor(np.zeros([BATCH_SIZE, 10]).astype(np.float32))
```

Finally, the network is wrapped with a Loss layer and an optimizer before being exported to a `MINDIR` file.

```python
from train_utils import TrainWrap
from mindspore.train.serialization import export
sgd=M.nn.SGD(trainable_weights, learning_rate=0.0015, momentum=0.9, dampening=0.01, weight_decay=0.0, nesterov=False, loss_scale=1.0)
net = TrainWrap(n, optimizer=sgd, weights=trainable_weights)
export(net, x, label, file_name="transfer_learning_tod.mindir", file_format='MINDIR')
```

Note that in this example we initialize the optimizer with only a few layers of trainable weights.
The provided TrainWrap function looks as follows:

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
  return train_net
```

To run this python code one must have an installed [MindSpore environment](https://gitee.com/mindspore/mindspore#installation). In the example below we use a GPU-supported MindSpore environment installed on a docker with image name `${DOCKER_IMG}`. It is passed to the script in the first parameter. Please refer to [MindSpore Docker Image Instalation instructions](https://gitee.com/mindspore/mindspore#docker-image). If a docker was not provided, the script will attempt to run the code locally

```bash
if [ -n "$1" ]; then
  DOCKER_IMG=$1
  docker run -w $PWD --runtime=nvidia -v /home/$USER:/home/$USER --privileged=true ${DOCKER_IMG} /bin/bash -c "python transfer_learning_export.py; chmod 444 transfer_learning_tod.mindir; rm -rf __pycache__"
else
  echo "MindSpore docker was not provided, attempting to run locally"
  python transfer_learning_export.py
fi
```

> MindSpore environment allows the developer to run MindSpore python code on server or PC. It differs from MindSpore Lite framework that allows to compile and run code on embedded devices.

## Converting the .mindir file into a ToD loadable .ms file

To convert the model simply use the converter as explained in the [Convert Section](https://www.mindspore.cn/tutorial/lite/en/master/use/create_model.html#converting-into-the-mindspore-tod-model)

```bash
CONVERTER="../../../../../mindspore/lite/build/tools/converter/converter_lite"
if [ ! -f "$CONVERTER" ]; then
    echo "$CONVERTER does not exist."
fi

echo "============Converting========="
$CONVERTER --fmk=MINDIR --trainModel=true --modelFile=transfer_learning_tod.mindir" --outputFile=transfer_learning_tod"
```

## Dataset

In this example we use a subset of the [Places dataset](http://places2.csail.mit.edu/) of scenes.
The whole dataset is composed of high resolution as well as small images and sums up to more than 100Gb.
For this demo we will use only the [validation data of small images](http://places2.csail.mit.edu/download.html) which is approximately 500Mb.

- Dataset size：501M，36,500 224*224 images in 365 classes
- Dataset format：jpg files
- In the current release, data is customely loaded using a proprietary DataSet class (provided in dataset.cc). In the upcoming releases loading will be done using MindSpore MindData infrastructure. In order to fit the data to the model it will be preprocessed using [ImageMagick convert tool](https://imagemagick.org/), namely croping and converting to bmp format
- Only 10 classes out of the 365 will be used in this demo
- 60% of the data will be used for training and 20% will be used for testing and the remaining 20% for validation

- The original dataset directory structure is as follows:

```text
places
├── val_256
│   ├── Places365_val_00000001.jpg
│   ├── Places365_val_00000002.jpg
│   ├── Places365_val_00000003.jpg
│   ├── Places365_val_00000004.jpg
│   ├── Places365_val_00000005.jpg
│   ├── .
│   ├── .
│   ├── .
│   ├── Places365_val_00036496.jpg
│   ├── Places365_val_00036497.jpg
│   ├── Places365_val_00036498.jpg
│   ├── Places365_val_00036499.jpg
│   └── Places365_val_00036500.jpg
```

The script [`prepare_dataset.sh`](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/example/transfer_learning/prepare_dataset.h) prepares the images to be used by the transfer learning model.
Only a subset of 10 classes (out of 365) are used in this example. Each image from these classes is center cropped to 224x224, additional channels are added in case of greyscale images, and the image is then converted to Bitmap format which is easier to read on the device.
In the upcoming releases, integration of MindSpore Lite framework with MindData framework will be available and will allow loading of `jpg` images and manipulating them on-Device.

## Implementing the main code and train loop

In the [example c++ code](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/tod/transfer_learning/src) the executable has the following API:

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

  float acc = CalculateAccuracy(ds_.val_data());
  std::cout << "accuracy = " << acc << std::endl;

  if (cycles_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained.ms";
    session_->SaveToFile(trained_fn);
  }
  return 0;
}
```

where `InitAndFigureInputs` creates the TrainSession instance from the `.ms` file, then sets the input tensors indices for this network as well as several derived variables

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

`InitDB` initializes the dataset from structured folders (each folder holds the images in BMP format) and loads it into the memory. We will not discuss this code here. During the load operation, the dataset is seperated into three sub-datasets. 60% of the images are inserted to the trainig dataset. 20% of them into the test dataset, and the remaining 20% are loaded to the validation dataset.
The user may refer to the [code in gitee](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/tod/transfer_learning/src/dataset.cc). In the next release, MindData framework will be integrated into this example.

The `TrainLoop` method is the core of the training procedure. We first display its code then review it.

```cpp
int NetRunner::TrainLoop() {
  session_->Train();
  float min_loss = 1000.;
  float max_acc = 0.;
  for (int i = 0; i < cycles_; i++) {
    FillInputData(ds_.train_data());
    session_->RunGraph(nullptr, verbose_ ? after_callback : nullptr);
    float loss = GetLoss();
    if (min_loss > loss) min_loss = loss;

    if (save_checkpoint_ != 0 && (i + 1) % save_checkpoint_ == 0) {
      auto cpkt_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained_" + std::to_string(i + 1) + ".ms";
      session_->SaveToFile(cpkt_fn);
    }

    std::cout << i + 1 << ": Loss is " << loss << " [min=" << min_loss << "]" << std::endl;
    if ((i + 1) % 20 == 0) {
      float acc = CalculateAccuracy(ds_.test_data());
      if (max_acc < acc) max_acc = acc;
      std::cout << "accuracy on test data = " << acc << " max accuracy = " << max_acc << std::endl;
      if (acc > 0.9) return 0;
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

Following the train cycle, the loss is [extracted from the Output Tensors](../use/use/runtime_cpp.html#obtaining-output-tensors).
It is advised to periodically save intermediate training results, i.e., checkpoint files. These files might be handy if the application or device crashes during the training process. The checkpoint files are practically `.ms` files that contain the updated weights, and the program may be relaunched with the checkpoint file as the `.ms` model file. Checkpoints are easily saved by calling the `SaveToFile` API, like this:

```cpp
  session_->SaveToFile(cpkt_file);
```

To keep track of the model accuracy, the `CalculateAccuracy` method is being called with the test dataset. Within which, the model is switched to `Eval` mode, and the method runs a cycle of test tensors through the trained network to measure the current accuracy rate. Since this method is time consuming it is not advised to run it every train cycle.

In the given example, the program terminates its run when `cycles_` training cycles have been finished or, when the accuracy on the test dataset reaches 90%. The user may easily change the termination condition, e.g., run only at night time when device is connected to a power source.

Finally, when trainining is completed, its accuracy is measured on the validation dataset and the fully trained model needs to be saved. The `SaveToFile` method is used for this purpose.

## Preparing and running

The code example provided in the [transfer learning directory](https://gitee.com/mindspore/mindspore/tree/master/lite/examples/transfer_learning/src/) includes a `prepare_and_run.sh` script that performs the followings:

- Prepare a folder that should be sent to the device
- Push this folder into the device
- Run the scripts on the device

The script accepts four paramaters:

- Places data directory.
- MindSpore docker image.
- A relaease tar file. Use the [downloaded](https://www.mindspore.cn/tutorial/lite/en/master/use/downloads.html) tar file or [compile mindspore for ToD](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#output-description).
- Target: arm64 for on-Device training or x86 for local testing.

### Preparing the model

Within the [model directory](https://gitee.com/mindspore/mindspore/tree/master/lite/examples/transfer_learning/model/) a `prepare_model.sh` script runs the python code that exports the model using a pre-prepared docker and then converts the model using the converter.

```bash
# Prepare the model
cd model/
rm -f *.ms
./prepare_model.sh $DOCKER
cd -
```

### Preparing the Folder

The `transfer_learning.ms` model file is then copied into the `package` folder as well as scripts and the MindSpore ToD library:

```bash
rm -rf ${PACKAGE}
mkdir -p ${PACKAGE}/model
cp model/*.ms ${PACKAGE}/model

# Copy the running script to the package
cp scripts/*.sh ${PACKAGE}/

# Copy the shared MindSpore ToD library
rm -rf msl
tar -xzf ${TARBALL} --wildcards --no-anchored libmindspore-lite.so
tar -xzf ${TARBALL} --wildcards --no-anchored include
mv mindspore-*/lib ${PACKAGE}/
mkdir -p msl
mv mindspore-*/* msl/
rm -rf mindspore-*
```

Then the dataset is converted from `jpg` format to `bmp`, cropped and organized in folders per class

```bash
# Convert the dataset into the package
./prepare_dataset.sh ${PLACES_DATA_PATH}
cp -r dataset ${PACKAGE}
```

### Compiling the code

Finally, the code is compiled for the traget and the binary is copied into the `package` folder.

```bash
# Compile program
make TARGET=${TARGET}

# Copy the executable to the package
mv bin ${PACKAGE}/
```

### Running the code on the device

The script then uses `adb` to push the code on to the device and run the set of scripts

```bash
# Push the folder to the device
echo "=======Pushing to device======="
adb push ${PACKAGE} /data/local/tmp/

echo "==Evaluating Untrained Model==="
adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval_untrained.sh"

echo "========Training on Device====="
adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh train.sh"

echo
echo "===Evaluating trained Model====="
adb shell "cd /data/local/tmp/package-arm64 && /system/bin/sh eval.sh"
echo
```

### Output

Running the script should take a few tens of seconds.
It slightly varies and depends on the randomly created weights in the network head.

The output on the device should look like this

```text
==Evaluating Untrained Model===
accuracy on validation data = 0.09375
========Training on Device=====
1: Loss is 5.02998 [min=5.02998]
2: Loss is 3.14241 [min=3.14241]
3: Loss is 2.11159 [min=2.11159]
4: Loss is 1.02237 [min=1.02237]
5: Loss is 1.49262 [min=1.02237]
6: Loss is 0.561587 [min=0.561587]
7: Loss is 0.602356 [min=0.561587]
8: Loss is 0.129092 [min=0.129092]
9: Loss is 0.525368 [min=0.129092]
10: Loss is 0.226777 [min=0.129092]
11: Loss is 0.294748 [min=0.129092]
12: Loss is 0.425187 [min=0.129092]
13: Loss is 1.12945 [min=0.129092]
14: Loss is 0.793843 [min=0.129092]
15: Loss is 1.1756 [min=0.129092]
16: Loss is 1.19964 [min=0.129092]
17: Loss is 0.274478 [min=0.129092]
18: Loss is 0.773159 [min=0.129092]
19: Loss is 0.3318 [min=0.129092]
20: Loss is 0.172462 [min=0.129092]
accuracy on test data = 0.885417 max accuracy = 0.885417
21: Loss is 1.06438 [min=0.129092]
22: Loss is 0.163407 [min=0.129092]
23: Loss is 0.59777 [min=0.129092]
24: Loss is 2.96935 [min=0.129092]
25: Loss is 0.98839 [min=0.129092]
26: Loss is 1.04646 [min=0.129092]
27: Loss is 0.0943134 [min=0.0943134]
28: Loss is 0.0167545 [min=0.0167545]
29: Loss is 0.108801 [min=0.0167545]
30: Loss is 0.240225 [min=0.0167545]
31: Loss is 0.376013 [min=0.0167545]
32: Loss is 0.0494523 [min=0.0167545]
33: Loss is 0.30756 [min=0.0167545]
34: Loss is 0.398682 [min=0.0167545]
35: Loss is 0.000215514 [min=0.000215514]
36: Loss is 0.465513 [min=0.000215514]
37: Loss is 0.433718 [min=0.000215514]
38: Loss is 0.253104 [min=0.000215514]
39: Loss is 0.00292511 [min=0.000215514]
40: Loss is 0.285255 [min=0.000215514]
accuracy on test data = 0.875 max accuracy = 0.885417
41: Loss is 0.112774 [min=0.000215514]
42: Loss is 0.929484 [min=0.000215514]
43: Loss is 0.479277 [min=0.000215514]
44: Loss is 0.0784799 [min=0.000215514]
45: Loss is 0.036647 [min=0.000215514]
46: Loss is 0.23905 [min=0.000215514]
47: Loss is 0.134072 [min=0.000215514]
48: Loss is 0.680484 [min=0.000215514]
49: Loss is 0.0302061 [min=0.000215514]
50: Loss is 0.156434 [min=0.000215514]
51: Loss is 0.133584 [min=0.000215514]
52: Loss is 0.00720471 [min=0.000215514]
53: Loss is 0.0276336 [min=0.000215514]
54: Loss is 0.000294939 [min=0.000215514]
55: Loss is 0.80308 [min=0.000215514]
56: Loss is 0.0460983 [min=0.000215514]
57: Loss is 0.237569 [min=0.000215514]
58: Loss is 0.00175199 [min=0.000215514]
59: Loss is 0.489163 [min=0.000215514]
60: Loss is 0.00449276 [min=0.000215514]
accuracy on test data = 0.895833 max accuracy = 0.895833
accuracy on validation data = 0.927083

===Evaluating trained Model=====
accuracy on validation data = 0.927083
```

In this example, a checkpoint file is saved every 20 iterations of the training

```text
(base) ~/mindspore/mindspore/lite/examples/transfer_learning$ adb shell ls -ltr /data/local/tmp/package-arm64/model
total 105660
-r-------- 1 root root 21612392 2019-01-06 22:11 transfer_learning_tod_trained_20.ms
-r-------- 1 root root 21612392 2019-01-06 22:18 transfer_learning_tod_trained_40.ms
-r-------- 1 root root 21612392 2019-01-06 22:25 transfer_learning_tod_trained_60.ms
-r-------- 1 root root 21612392 2019-01-06 22:28 transfer_learning_tod_trained.ms
-r--r--r-- 1 root root 21612392 2020-12-03 09:08 transfer_learning_tod.ms
```
