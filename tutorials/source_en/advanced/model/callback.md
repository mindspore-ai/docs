# Callback Mechanism

<a href="https://gitee.com/mindspore/docs/blob/r1.10/tutorials/source_en/advanced/model/callback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

During deep learning training, MindSpore provides the callback mechanism to promptly learn about the training status of the network model, observe the changes of network model parameters in real time, and implement customized operations during training.

The callback mechanism is generally used in the network model training process `model.train`. The MindSpore `model` executes callback functions based on the sequence in the callback list. You can set different callback classes to implement functions executed during or after training.

> For more information about built-in callback classes and how to use them, see [API](https://www.mindspore.cn/docs/en/r1.10/api_python/mindspore/mindspore.Callback.html#mindspore.Callback).

## Callback Usage

When talking about callback, most users find it difficult to understand whether stacks or special scheduling modes are required. Actually, the callback can be explained as follows:

Assume that function A has a parameter which is function B. After function A is executed, function B is executed. This process is called callback.

The `callback` in MindSpore is actually not a function but a class. You can use the callback mechanism to observe the internal status and related information of the network during training or perform specific actions in a specific period.

For example, monitor the loss function, save the model parameter `ckpt`, dynamically adjust the parameter `lr`, and terminate the training task in advance.

The following uses the LeNet-5 model training based on the MNIST dataset as an example to describe several common MindSpore built-in callback classes.

Download and process MNIST data to build a LeNet-5 model. The sample code is as follows:

```python
from download import download
from mindspore import nn, Model
from mindspore.dataset import vision, transforms, MnistDataset
from mindspore.common.initializer import Normal
from mindspore import dtype as mstype

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"

# download dataset
download(url, "./", kind="zip", replace=True)

# process dataset
def proc_dataset(data_path, batch_size=32):
    mnist_ds = MnistDataset(data_path, shuffle=True)

    # define map operations
    image_transforms = [
        vision.Resize(32),
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]

    label_transform = transforms.TypeCast(mstype.int32)

    mnist_ds = mnist_ds.map(operations=label_transform, input_columns="label")
    mnist_ds = mnist_ds.map(operations=image_transforms, input_columns="image")
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    return mnist_ds

train_dataset = proc_dataset('MNIST_Data/train')

# define LeNet-5 model
class LeNet5(nn.Cell):

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_model():
    model = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(model.trainable_params(), learning_rate=0.01, momentum=0.9)
    trainer = Model(model, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": nn.Accuracy()})
    return trainer

trainer = create_model()
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:01<00:00, 10.0MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

## Common Built-in Callback Functions

MindSpore provides the `callback` capability to allow users to insert customized operations in a specific phase of training or inference.

### ModelCheckpoint

To save the trained network model and parameters for re-inference or re-training, MindSpore provides the [ModelCheckpoint](https://mindspore.cn/docs/en/r1.10/api_python/mindspore/mindspore.ModelCheckpoint.html#mindspore.ModelCheckpoint) API, which is generally used together with the [CheckpointConfig](https://mindspore.cn/docs/en/r1.10/api_python/mindspore/mindspore.CheckpointConfig.html#mindspore.CheckpointConfig) API.

```python
from mindspore import CheckpointConfig, ModelCheckpoint

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
├── mnist-1_1875.ckpt # file to save parameters
└── mnist-graph.meta # grapg after compiled
```

### LossMonitor

To monitor the change of the loss function value during training, set `per_print_times` to control the interval of printing loss.

```python
from mindspore import LossMonitor

loss_monitor = LossMonitor(1875)
# Start training and load the saved model and parameter callback function.
trainer.train(3, train_dataset, callbacks=[loss_monitor])
```

```text
epoch: 1 step: 1875, loss is 0.008795851841568947
epoch: 2 step: 1875, loss is 0.007240554317831993
epoch: 3 step: 1875, loss is 0.0036914246156811714
```

During training, LossMonitor monitors the loss value of training. And when you train and infer at the same time, LossMonitor monitors the loss value of training and the Metrics value of inferring.

```python
test_dataset = proc_dataset('MNIST_Data/test')

trainer.fit(2, train_dataset, test_dataset, callbacks=[loss_monitor])
```

```text
epoch: 1 step: 1875, loss is 0.0026960039976984262
Eval result: epoch 1, metrics: {'Accuracy': 0.9888822115384616}
epoch: 2 step: 1875, loss is 0.00038617433165200055
Eval result: epoch 2, metrics: {'Accuracy': 0.9877804487179487}
```

### TimeMonitor

To monitor the execution time of training or testing, set `data_size` to control the interval of printing the execution time.

```python
from mindspore import TimeMonitor

time_monitor = TimeMonitor(1875)
trainer.train(1, train_dataset, callbacks=[time_monitor])
```

```text
Train epoch time: 3876.302 ms, per step time: 2.067 ms
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
from mindspore import Callback

class StopTimeMonitor(Callback):

    def __init__(self, run_time):
        """Define the initialization process."""
        super(StopTimeMonitor, self).__init__()
        self.run_time = run_time             # Define the execution time.

    def on_train_begin(self, run_context):
        """Operations when training is started."""
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

train_dataset = proc_dataset('MNIST_Data/train')
trainer.train(10, train_dataset, callbacks=[LossMonitor(), StopTimeMonitor(4)])
```

```text
Begin training, time is: 1673515004.6783535
epoch: 1 step: 1875, loss is 0.0006050781812518835
End training, time: 1673515009.1824663, epoch: 1, step: 1875, loss:0.0006050782
```

According to the preceding information, the progrem stopped immediately the running time reaches the threshold.

### Customized Model Saving Threshold

This callback mechanism is used to save the network model weight CKPT file when the loss is less than the specified threshold.

The sample code is as follows:

```python
from mindspore import save_checkpoint

# Define the callback API for saving the CKPT file.
class SaveCkptMonitor(Callback):
    """Define the initialization process."""

    def __init__(self, loss):
        super(SaveCkptMonitor, self).__init__()
        self.loss = loss  # Defines the loss threshold.

    def on_train_step_end(self, run_context):
        """Define the operation to be performed when a step ends."""
        cb_params = run_context.original_args()
        cur_loss = cb_params.net_outputs.asnumpy() # Obtain the current loss value.

        # If the current loss value is less than the preset threshold, the training stops.
        if cur_loss < self.loss:
            # Name the file to be saved.
            file_name = f"./checkpoint/{cb_params.cur_epoch_num}_{cb_params.cur_step_num}.ckpt"
            # Save the network model.
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Saved checkpoint, loss:{:8.7f}, current step num:{:4}.".format(cur_loss, cb_params.cur_step_num))

trainer = create_model()
train_dataset = proc_dataset('MNIST_Data/train')
trainer.train(5, train_dataset, callbacks=[LossMonitor(), SaveCkptMonitor(0.01)])
```

```text
epoch: 1 step: 1875, loss is 0.15191984176635742
epoch: 2 step: 1875, loss is 0.14701086282730103
epoch: 3 step: 1875, loss is 0.0020134493242949247
Saved checkpoint, loss:0.0020134, current step num:5625.
epoch: 4 step: 1875, loss is 0.018305214121937752
epoch: 5 step: 1875, loss is 0.00019801077723968774
Saved checkpoint, loss:0.0001980, current step num:9375.
```

Finally, the network weights whose loss value is less than the threshold is saved in `./checkpoint/` directory.