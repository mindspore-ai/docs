# Fault Recovery

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/train_availability/fault_recover.md)

## Overview

Faults may be encountered during model training. The overhead of restarting the training with various resources is huge. For this purpose, MindSpore provides a fault recovery scheme, i.e., periodically saving the model parameters, which allows the model to recover quickly and continue training at the point of failure.

MindSpore saves the model parameters in a step or epoch cycle. The model parameters are saved in Checkpoint (ckpt for short) files. If a fault occurs during model training, load the latest saved model parameters, restore the state here, and continue training.

> This document describes the use case for fault recovery, saving the Checkpoint file only at the end of each epoch.

## Data and Model Preparation

To provide a complete experience, the fault recovery process is simulated here by using the MNIST dataset and the LeNet5 network. You can skip this section if you are ready.

### Data Preparation

Download the MNIST dataset and unzip the dataset to the project directory.

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
```

### Model Definition

```python
import os

import mindspore
from mindspore.common.initializer import Normal
from mindspore.dataset import MnistDataset, vision
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, Callback
import mindspore.dataset.transforms as transforms

mindspore.set_context(mode=mindspore.GRAPH_MODE)


# Create a training dataset
def create_dataset(data_path, batch_size=32):
    train_dataset = MnistDataset(data_path, shuffle=False)
    image_transfroms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Resize(size=(32, 32)),
        vision.HWC2CHW()
    ]
    train_dataset = train_dataset.map(image_transfroms, input_columns='image')
    train_dataset = train_dataset.map(transforms.TypeCast(mindspore.int32), input_columns='label')
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    return train_dataset


# Load the training dataset
data_path = "MNIST_Data/train"
train_dataset = create_dataset(data_path)

# Fault during the simulation training
class myCallback(Callback):
    def __init__(self, break_epoch_num=6):
        super(myCallback, self).__init__()
        self.epoch_num = 0
        self.break_epoch_num = break_epoch_num

    def on_train_epoch_end(self, run_context):
        self.epoch_num += 1
        if self.epoch_num == self.break_epoch_num:
            raise Exception("Some errors happen.")


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet5()  # Model initialization
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")  # Loss function
optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)  # Optimizer
model = Model(net, loss_fn=loss, optimizer=optim)  # Model encapsulation
```

## Periodically Saving Checkpoint Files

### Configuring CheckpointConfig

`mindspore.train.CheckpointConfig` supports configuration based on the number of iterations, with the following main parameters:

- `save_checkpoint_steps`: indicates how many steps to save a Checkpoint file. The default value is 1.
- `keep_checkpoint_max`: indicates the maximum number of Checkpoint files to be saved. The default value is 5.

If the iteration strategy script ends normally, the Checkpoint file of the last step is saved by default.

During model training, the `callbacks` parameter in `Model.train` is used to pass in the object `ModelCheckpoint` of saving model (used in conjunction with `mindspore.train.CheckpointConfig`), which generates Checkpoint file.

### User-defined Saved Data

The parameter `append_info` of `CheckpointConfig` can save user-defined information in the Checkpoint file. `append_info` supports passing in ``epoch_num``, ``step_num`` and data of dictionary type. ``epoch_num`` and ``step_num`` can save the number of epochs and the number of steps during training in the Checkpoint file.
`key` of the dictionary type data must be of type string, and `value` must be of type int, float, bool, string, Parameter, or Tensor.

```python
# User-defined saved data
append_info = ["epoch_num", "step_num", {"lr": 0.01, "momentum": 0.9}]
# In the data sinking mode, the Checkpoint file of the last step is saved by default
config_ck = CheckpointConfig(append_info=append_info)
# The Checkpoint file is saved with the prefix "lenet" and is saved in "./lenet" path
ckpoint_cb = ModelCheckpoint(prefix='lenet', directory='./lenet', config=config_ck)

# Simulation program fault. The default is to fail at the end of the 6th epoch
my_callback = myCallback()

# In the data sinking mode, 10 epoch training is performed by using Model.train
model.train(10, train_dataset, callbacks=[ckpoint_cb, my_callback], dataset_sink_mode=True)
```

## User-defined Script to Find the Latest Checkpoint File

The program fails at the end of the 6th epoch. After the failure, the `./lenet` directory holds the Checkpoint files for the latest generated 5 epochs.

```text
└── lenet
     ├── lenet-graph.meta  # Compiled compute graph
     ├── lenet-2_1875.ckpt  # Checkpoint files with the suffix '.ckpt'
     ├── lenet-3_1875.ckpt  # The naming of the file indicates the number of epochs and steps where the parameters are stored. Here is the model parameters of the 1875th step of the 3rd epoch
     ├── lenet-4_1875.ckpt
     ├── lenet-5_1875.ckpt
     └── lenet-6_1875.ckpt
```

> If the user runs the training script multiple times using the same prefix name, a Checkpoint file with the same name may be generated. MindSpore adds "_" and a number after the user-defined prefix to make it easier for users to distinguish between the files generated each time. If you want to delete the .ckpt file, please delete the .meta file at the same time. For example: `lenet_3-2_1875.ckpt` indicates the Checkpoint file for the 1875th step of the 2nd epoch generated by running the fourth script.

Users can use user-defined scripts to find the latest saved Checkpoint files.

```python
ckpt_path = "./lenet"
filenames = os.listdir(ckpt_path)
# Filter all Checkpoint file names
ckptnames = [ckpt for ckpt in filenames if ckpt.endswith(".ckpt")]
# Sort Checkpoint file names from oldest to newest in order of creation
ckptnames.sort(key=lambda ckpt: os.path.getctime(ckpt_path + "/" + ckpt))
# Get the latest Checkpoint file path
ckpt_file = ckpt_path + "/" + ckptnames[-1]
```

## Recovery Training

### Loading Checkpoint File

Use the `load_checkpoint` and `load_param_into_net` methods to load the latest saved Checkpoint file.

- The `load_checkpoint` method will load the network parameters from the Checkpoint file into the dictionary param_dict.
- The `load_param_into_net` method will load the parameters from the dictionary param_dict into the network or optimizer, and the parameters in the network after loading are the ones saved in the Checkpoint file.

```python
# Load the model parameters into param_dict. Here the model parameters saved during training and the user-defined saved data are loaded
param_dict = mindspore.load_checkpoint(ckpt_file)
net = LeNet5()
# Load the parameters into the model
mindspore.load_param_into_net(net, param_dict)
```

### Obtaining the User-defined Data

The user can obtain the number of epochs and user-defined saved data from the Checkpoint file for training. Note that the data obtained at this point is of type Parameter.

```python
epoch_num = int(param_dict["epoch_num"].asnumpy())
step_num = int(param_dict["step_num"].asnumpy())
lr = float(param_dict["lr"].asnumpy())
momentum = float(param_dict["momentum"].asnumpy())
```

### Setting the Epoch for Continued Training

Pass the number of obtained epochs to the `initial_epoch` parameter of `Model.train`. The network will continue training from that epoch. In this case, the `epoch` parameter of `Model.train` indicates the last epoch of training.

```python
model.train(10, train_dataset, callbacks=ckpoint_cb, initial_epoch=epoch_num, dataset_sink_mode=True)
```

### Training Ends

At the end of the training, `./lenet` directory generates 4 new Checkpoint files. Based on the Checkpoint file names, it can be seen that the model is retrained at the 7th epoch and ends at the 10th epoch after the failure occurs. The fault recovery is successful.

```text
└── lenet
     ├── lenet-graph.meta
     ├── lenet-2_1875.ckpt
     ├── lenet-3_1875.ckpt
     ├── lenet-4_1875.ckpt
     ├── lenet-5_1875.ckpt
     ├── lenet-6_1875.ckpt
     ├── lenet-1-7_1875.ckpt
     ├── lenet-1-8_1875.ckpt
     ├── lenet-1-9_1875.ckpt
     ├── lenet-1-10_1875.ckpt
     └── lenet-1-graph.meta
```
