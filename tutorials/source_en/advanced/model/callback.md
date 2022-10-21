# Callback Mechanism

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/model/callback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

During deep learning training, MindSpore provides the callback mechanism to promptly learn about the training status of the network model, observe the changes of network model parameters in real time, and implement customized operations during training.

The callback mechanism is generally used in the network model training process `model.train`. The MindSpore `model` executes callback functions based on the sequence in the callback list. You can set different callback classes to implement functions executed during or after training.

> For more information about built-in callback classes and how to use them, see [API](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Callback.html#mindspore.train.Callback).

## Callback Introduction

When talking about callback, most users find it difficult to understand whether stacks or special scheduling modes are required. Actually, the callback can be explained as follows:

Assume that function A has a parameter which is function B. After function A is executed, function B is executed. This process is called callback.

The `callback` in MindSpore is actually not a function but a class. You can use the callback mechanism to observe the internal status and related information of the network during training or perform specific actions in a specific period.

For example, monitor the loss function, save the model parameter `ckpt`, dynamically adjust the parameter `lr`, and terminate the training task in advance. The following uses the MNIST dataset as an example to describe several common built-in callback functions and customised callback functions.

```python
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore.train import Model

# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:01<00:00, 10.0MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

```python
train_dataset = datapipe('MNIST_Data/train', 64)
test_dataset = datapipe('MNIST_Data/test', 64)

trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
```

## Common Built-in Callback Functions

MindSpore provides the `callback` capability to allow users to insert customized operations in a specific phase of training or inference.

### ModelCheckpoint

To save the trained network model and parameters for re-inference or re-training, MindSpore provides the [ModelCheckpoint](https://mindspore.cn/docs/en/master/api_python/train/mindspore.train.ModelCheckpoint.html#mindspore.train.ModelCheckpoint) API, which is generally used together with the [CheckpointConfig](https://mindspore.cn/docs/en/master/api_python/train/mindspore.train.CheckpointConfig.html#mindspore.train.CheckpointConfig) API.

```python
from mindspore.train import ModelCheckpoint, CheckpointConfig

# Set the configuration information of the saved model.
config = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# Instantiate the saved model callback API and define the storage path and prefix.
ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)

# Start training and load the saved model and parameter callback function.
trainer.train(1, train_dataset, callbacks=[ckpt_callback])
```

After the preceding code is executed, the generated checkpoint file directory structure is as follows:

```text
./checkpoint/
├── mnist-1_938.ckpt # file to save parameters
└── mnist-graph.meta # grapg after compiled
```

### LossMonitor

To monitor the change of the loss function value during training, set `per_print_times` to control the interval of printing loss.

```python
from mindspore.train import LossMonitor

loss_monitor = LossMonitor(300)
# Start training and load the saved model and parameter callback function. The input parameters of LossMonitor are learning rate (0.01) and stride (375).
trainer.train(1, train_dataset, callbacks=[loss_monitor])
```

```text
    epoch: 1 step: 300, loss is 0.45305341482162476
    epoch: 1 step: 600, loss is 0.2915695905685425
    epoch: 1 step: 900, loss is 0.5174192190170288
```

During training, LossMonitor monitors the loss value of training. And when you train and infer at the same time, LossMonitor monitors the loss value of training and the Metrics value of inferring.

```python
trainer.fit(1, train_dataset, test_dataset, callbacks=[loss_monitor])
```

```text
    epoch: 1 step: 300, loss is 0.3167177438735962
    epoch: 1 step: 600, loss is 0.36215940117836
    epoch: 1 step: 900, loss is 0.25714176893234253
    Eval result: epoch 1, metrics: {'accuracy': 0.9202}
```

### TimeMonitor

To monitor the execution time of training or testing, set `data_size` to control the interval of printing the execution time.

```python
from mindspore.train import TimeMonitor

time_monitor = TimeMonitor()
trainer.train(1, train_dataset, callbacks=[time_monitor])
```

```text
Train epoch time: 7388.254 ms, per step time: 7.877 ms
```

## Customized Callback Mechanism

MindSpore not only has powerful built-in callback functions, but also allows users to customize callback classes based on the `Callback` base class when they have special requirements.

You can customize callbacks based on the `Callback` base class as required. The `Callback` base class is defined as follows:

```python
class Callback():
    """Callback base class"""
    def on_train_begin(self, run_context):
        """Called once before the network executing."""

    def on_train_epoch_begin(self, run_context):
        """Called before each epoch beginning."""

    def on_train_epoch_end(self, run_context):
        """Called after each epoch finished."""

    def on_train_step_begin(self, run_context):
        """Called before each step beginning."""

    def on_train_step_end(self, run_context):
        """Called after each step finished."""

    def on_train_end(self, run_context):
        """Called once after network training."""
```

The callback mechanism can record important information during training and transfer a dictionary variable `RunContext.original_args()` to the callback object so that users can obtain related attributes from each customized callback, perform customized operations, and customize other variables and transfer them to the `RunContext.original_args()` object.

Common attributes in `RunContext.original_args()` are as follows:

- epoch_num: number of training epochs
- batch_num: number of steps in an epoch
- cur_epoch_num: number of current epochs
- cur_step_num: number of current steps

- loss_fn: loss function
- optimizer: optimizer
- train_network: training network
- train_dataset: training dataset
- net_outputs: network output

- parallel_mode: parallel mode
- list_callback: all callback functions

You can understand the customized callback mechanism in the following two scenarios:

### Customized Training Termination Time

The training can be terminated within a specified period. You can set a time threshold. When the training time reaches the threshold, the training process is terminated.

In the following code, the `run_context.original_args` method can be used to obtain the `cb_params` dictionary which contains the main attribute information described above.

In addition, you can modify and add values in the dictionary. Define an `init_time` object in the `begin` function and transfer it to the `cb_params` dictionary. After each step ends, the system checks whether the training time is greater than the configured time threshold. If the training time is greater than the configured time threshold, the system sends a training termination signal to `run_context` to terminate the training in advance and prints the current epoch, step, and loss values.

```python
import time
import mindspore as ms

class StopTimeMonitor(ms.train.Callback):

    def __init__(self, run_time):
        """Define the initialization process."""
        super(StopTimeMonitor, self).__init__()
        self.run_time = run_time            # Define the execution time.

    def on_train_begin(self, run_context):
        """Operations when training is started.""
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()   # Obtain the current timestamp as the training start time.
        print(f"Begin training, time is: {cb_params.init_time}")

    def on_train_step_end(self, run_context):
        """Operations after each step ends."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num  # Obtain the epoch value.
        step_num = cb_params.cur_step_num    # Obtain the step value.
        loss = cb_params.net_outputs         # Obtain the loss value.
        cur_time = time.time()               # Obtain the current timestamp.

        if (cur_time - cb_params.init_time) > self.run_time:
            print(f"End training, time: {cur_time}, epoch: {epoch_num}, step: {step_num}, loss:{loss}")
            run_context.request_stop()       # Stop training.

datasize = train_dataset.get_dataset_size()
trainer.train(5, train_dataset, callbacks=[LossMonitor(datasize), StopTimeMonitor(4)])
```

```text
    Begin training, time is: 1665892816.363511
    End training, time: 1665892820.3696215, epoch: 1, step: 575, loss:Tensor(shape=[], dtype=Float32, value= 0.35758)
```

According to the preceding information, when step 4673 of the third epoch is complete, the running time reaches the threshold and the training ends.

### Customized Model Saving Threshold

This callback mechanism is used to save the network model weight CKPT file when the loss is less than the specified threshold.

The sample code is as follows:

```python
import mindspore as ms

# Define the callback API for saving the CKPT file.
class SaveCkptMonitor(ms.train.Callback):
    """Define the initialization process."""

    def __init__(self, loss):
        super(SaveCkptMonitor, self).__init__()
        self.loss = loss # Defines the loss threshold.

    def on_train_step_end(self, run_context):
        """Define the operation to be performed when a step ends."""
        cb_params = run_context.original_args()
        cur_loss = cb_params.net_outputs.asnumpy() # Obtain the current loss value.

        # If the current loss value is less than the preset threshold, the training stops.
        if cur_loss < self.loss:
            # Name the file to be saved.
            file_name = f"./checkpoint/{cb_params.cur_epoch_num}_{cb_params.cur_step_num}.ckpt"
            # Save the network model.
            ms.save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Saved checkpoint, loss:{:8.7f}, current step num:{:4}.".format(cur_loss, cb_params.cur_step_num))

trainer.train(1, train_dataset, callbacks=[SaveCkptMonitor(0.01)])
```

The directory structure is as follows:

```text
./
├── 1_253.ckpt
├── 1_258.ckpt
├── 1_265.ckpt
├── 1_332.ckpt
├── 1_358.ckpt
├── 1_380.ckpt
├── 1_395.ckpt
├── 1_1151.ckpt
├── 1_1358.ckpt
├── 1_1524.ckpt
```
