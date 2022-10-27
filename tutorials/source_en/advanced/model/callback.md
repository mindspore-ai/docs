# Callback Mechanism

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/advanced/model/callback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

During deep learning training, MindSpore provides the callback mechanism to promptly learn about the training status of the network model, observe the changes of network model parameters in real time, and implement customized operations during training.

The callback mechanism is generally used in the network model training process `model.train`. The MindSpore `model` executes callback functions based on the sequence in the callback list. You can set different callback classes to implement functions executed during or after training.

> For more information about built-in callback classes and how to use them, see [API](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.Callback.html#mindspore.Callback).

## Callback Usage

When talking about callback, most users find it difficult to understand whether stacks or special scheduling modes are required. Actually, the callback can be explained as follows:

Assume that function A has a parameter which is function B. After function A is executed, function B is executed. This process is called callback.

The `callback` in MindSpore is actually not a function but a class. You can use the callback mechanism to observe the internal status and related information of the network during training or perform specific actions in a specific period.

For example, monitor the loss function, save the model parameter `ckpt`, dynamically adjust the parameter `lr`, and terminate the training task in advance.

The following uses the LeNet-5 model training based on the MNIST dataset as an example to describe several common MindSpore built-in callback classes.

Download and process MNIST data to build a LeNet-5 model. The sample code is as follows:

```python
import mindspore.nn as nn
import mindspore as ms
from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet

download_train = Mnist(path="./mnist", split="train", download=True)
dataset_train = download_train.run()

network = lenet(num_classes=10, pretrained=False)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)

# Define a network model.
model = ms.Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": nn.Accuracy()})
```

To use the callback mechanism, transfer the `callback` object to the `model.train` method. The `callback` object can be a callback list. The sample code is as follows, where [ModelCheckpoint](https://mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.ModelCheckpoint.html#mindspore.ModelCheckpoint) and [LossMonitor](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.LossMonitor.html#mindspore.LossMonitor) are callback classes provided by MindSpore:

```python
import mindspore as ms

# Define callback classes.
ckpt_cb = ms.ModelCheckpoint()
loss_cb = ms.LossMonitor(1875)

model.train(5, dataset_train, callbacks=[ckpt_cb, loss_cb])
```

```text
    epoch: 1 step: 1875, loss is 0.257398396730423
    epoch: 2 step: 1875, loss is 0.04801357910037041
    epoch: 3 step: 1875, loss is 0.028765171766281128
    epoch: 4 step: 1875, loss is 0.008372672833502293
    epoch: 5 step: 1875, loss is 0.0016194271156564355
```

## Common Built-in Callback Functions

MindSpore provides the `callback` capability to allow users to insert customized operations in a specific phase of training or inference.

### ModelCheckpoint

To save the trained network model and parameters for re-inference or re-training, MindSpore provides the [ModelCheckpoint](https://mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.ModelCheckpoint.html#mindspore.ModelCheckpoint) API, which is generally used together with the [CheckpointConfig](https://mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.CheckpointConfig.html#mindspore.CheckpointConfig) API.

The following uses a sample code to describe how to save the trained network model and parameters.

```python
import mindspore as ms

# Set the configuration information of the saved model.
config_ck = ms.CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# Instantiate the saved model callback API and define the storage path and prefix.
ckpoint = ms.ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)

# Start training and load the saved model and parameter callback function.
model.train(1, dataset_train, callbacks=[ckpoint])
```

After the preceding code is executed, the generated checkpoint file directory structure is as follows:

```text
./lenet/
├── lenet-1_1875.ckpt # Parameter file.
└── lenet-graph.meta # Computational graph after compiled.
```

### LossMonitor

To monitor the change of the loss function value during training and observe the running time of each epoch and step during training, [MindSpore Vision](https://mindspore.cn/vision/docs/en/master/index.html) provides the `LossMonitor` API (different from the `LossMonitor` API provided by MindSpore).

The following uses sample code as an example:

```python
from mindvision.engine.callback import LossMonitor

# Start training and load the saved model and parameter callback function. The input parameters of LossMonitor are learning rate (0.01) and stride (375).
model.train(5, dataset_train, callbacks=[LossMonitor(0.01, 375)])
```

```text
    Epoch:[  0/  5], step:[  375/ 1875], loss:[0.041/0.023], time:0.670 ms, lr:0.01000
    Epoch:[  0/  5], step:[  750/ 1875], loss:[0.002/0.023], time:0.723 ms, lr:0.01000
    Epoch:[  0/  5], step:[ 1125/ 1875], loss:[0.006/0.023], time:0.662 ms, lr:0.01000
    Epoch:[  0/  5], step:[ 1500/ 1875], loss:[0.000/0.024], time:0.664 ms, lr:0.01000
    Epoch:[  0/  5], step:[ 1875/ 1875], loss:[0.009/0.024], time:0.661 ms, lr:0.01000
    Epoch time: 1759.622 ms, per step time: 0.938 ms, avg loss: 0.024
    Epoch:[  1/  5], step:[  375/ 1875], loss:[0.001/0.020], time:0.658 ms, lr:0.01000
    Epoch:[  1/  5], step:[  750/ 1875], loss:[0.002/0.021], time:0.661 ms, lr:0.01000
    Epoch:[  1/  5], step:[ 1125/ 1875], loss:[0.000/0.021], time:0.663 ms, lr:0.01000
    Epoch:[  1/  5], step:[ 1500/ 1875], loss:[0.048/0.022], time:0.655 ms, lr:0.01000
    Epoch:[  1/  5], step:[ 1875/ 1875], loss:[0.018/0.022], time:0.646 ms, lr:0.01000
    Epoch time: 1551.506 ms, per step time: 0.827 ms, avg loss: 0.022
    Epoch:[  2/  5], step:[  375/ 1875], loss:[0.001/0.017], time:0.674 ms, lr:0.01000
    Epoch:[  2/  5], step:[  750/ 1875], loss:[0.001/0.018], time:0.669 ms, lr:0.01000
    Epoch:[  2/  5], step:[ 1125/ 1875], loss:[0.004/0.019], time:0.683 ms, lr:0.01000
    Epoch:[  2/  5], step:[ 1500/ 1875], loss:[0.003/0.020], time:0.657 ms, lr:0.01000
    Epoch:[  2/  5], step:[ 1875/ 1875], loss:[0.041/0.019], time:1.447 ms, lr:0.01000
    Epoch time: 1616.589 ms, per step time: 0.862 ms, avg loss: 0.019
    Epoch:[  3/  5], step:[  375/ 1875], loss:[0.000/0.011], time:0.672 ms, lr:0.01000
    Epoch:[  3/  5], step:[  750/ 1875], loss:[0.001/0.013], time:0.687 ms, lr:0.01000
    Epoch:[  3/  5], step:[ 1125/ 1875], loss:[0.016/0.014], time:0.665 ms, lr:0.01000
    Epoch:[  3/  5], step:[ 1500/ 1875], loss:[0.001/0.015], time:0.674 ms, lr:0.01000
    Epoch:[  3/  5], step:[ 1875/ 1875], loss:[0.001/0.015], time:0.666 ms, lr:0.01000
    Epoch time: 1586.809 ms, per step time: 0.846 ms, avg loss: 0.015
    Epoch:[  4/  5], step:[  375/ 1875], loss:[0.000/0.008], time:0.671 ms, lr:0.01000
    Epoch:[  4/  5], step:[  750/ 1875], loss:[0.000/0.013], time:0.701 ms, lr:0.01000
    Epoch:[  4/  5], step:[ 1125/ 1875], loss:[0.009/0.015], time:0.666 ms, lr:0.01000
    Epoch:[  4/  5], step:[ 1500/ 1875], loss:[0.008/0.015], time:0.941 ms, lr:0.01000
    Epoch:[  4/  5], step:[ 1875/ 1875], loss:[0.008/0.015], time:0.661 ms, lr:0.01000
    Epoch time: 1584.785 ms, per step time: 0.845 ms, avg loss: 0.015
```

According to the preceding information, the information printed by the `LossMonitor` API provided by the [MindSpore Vision toolkit](https://mindspore.cn/vision/docs/en/master/index.html) is more detailed. The stride is set to 375. Therefore, one record is printed every 375 steps, and the loss value fluctuates. However, in general, the loss value decreases gradually and the accuracy increases gradually.

### ValAccMonitor

To save the network model and parameters with the optimal accuracy during training, you need to validate them while training. MindSpore Vision provides the `ValAccMonitor` API.

The following uses an example to describe the process.

```python
from mindvision.engine.callback import ValAccMonitor

download_eval = Mnist(path="./mnist", split="test", download=True)
dataset_eval = download_eval.run()

# Start training and load the saved model and parameter callback function.
model.train(1, dataset_train, callbacks=[ValAccMonitor(model, dataset_eval, num_epochs=1)])
```

```text
    --------------------
    Epoch: [  1 /   1], Train Loss: [0.000], Accuracy:  0.988
    ================================================================================
    End of validation the best Accuracy is:  0.988, save the best ckpt file in ./best.ckpt
```

After the preceding code is executed, the network model and parameters with the optimal accuracy are saved as the `best.ckpt` file in the current directory.

## Customized Callback Mechanism

MindSpore not only has powerful built-in callback functions, but also allows users to customize callback classes based on the `Callback` base class when they have special requirements.

You can customize callbacks based on the `Callback` base class as required. The `Callback` base class is defined as follows:

```python
class Callback():
    """Callback base class"""
    def begin(self, run_context):
        """Called once before the network executing."""
        pass # pylint: disable=W0107

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        pass # pylint: disable=W0107

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        pass # pylint: disable=W0107

    def step_begin(self, run_context):
        """Called before each step beginning."""
        pass # pylint: disable=W0107

    def step_end(self, run_context):
        """Called after each step finished."""
        pass # pylint: disable=W0107

    def end(self, run_context):
        """Called once after network training."""
        pass # pylint: disable=W0107
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

class StopTimeMonitor(ms.Callback):

    def __init__(self, run_time):
        """Define the initialization process."""
        super(StopTimeMonitor, self).__init__()
        self.run_time = run_time            # Define the execution time.

    def begin(self, run_context):
        """Operations when training is started.""
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()   # Obtain the current timestamp as the training start time.
        print("Begin training, time is:", cb_params.init_time)

    def step_end(self, run_context):
        """Operations after each step ends."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num  # Obtain the epoch value.
        step_num = cb_params.cur_step_num    # Obtain the step value.
        loss = cb_params.net_outputs         # Obtain the loss value.
        cur_time = time.time()               # Obtain the current timestamp.

        if (cur_time - cb_params.init_time) > self.run_time:
            print("End training, time:", cur_time, ",epoch:", epoch_num, ",step:", step_num, ",loss:", loss)
            run_context.request_stop()       # Stop training.

download_train = Mnist(path="./mnist", split="train", download=True)
dataset = download_train.run()
model.train(5, dataset, callbacks=[LossMonitor(0.01, 1875), StopTimeMonitor(4)])
```

```text
    Begin training, time is: 1648452437.2004516
    Epoch:[  0/  5], step:[ 1875/ 1875], loss:[0.011/0.012], time:0.678 ms, lr:0.01000
    Epoch time: 1603.104 ms, per step time: 0.855 ms, avg loss: 0.012
    Epoch:[  1/  5], step:[ 1875/ 1875], loss:[0.000/0.011], time:0.688 ms, lr:0.01000
    Epoch time: 1602.716 ms, per step time: 0.855 ms, avg loss: 0.011
    End training, time: 1648452441.20081 ,epoch: 3 ,step: 4673 ,loss: 0.014888153
    Epoch time: 792.901 ms, per step time: 0.423 ms, avg loss: 0.010
```

According to the preceding information, when step 4673 of the third epoch is complete, the running time reaches the threshold and the training ends.

### Customized Model Saving Threshold

This callback mechanism is used to save the network model weight CKPT file when the loss is less than the specified threshold.

The sample code is as follows:

```python
import mindspore as ms

# Define the callback API for saving the CKPT file.
class SaveCkptMonitor(ms.Callback):
    """Define the initialization process."""

    def __init__(self, loss):
        super(SaveCkptMonitor, self).__init__()
        self.loss = loss # Defines the loss threshold.

    def step_end(self, run_context):
        """Define the operation to be performed when a step ends."""
        cb_params = run_context.original_args()
        cur_loss = cb_params.net_outputs.asnumpy() # Obtain the current loss value.

        # If the current loss value is less than the preset threshold, the training stops.
        if cur_loss < self.loss:
            # Name the file to be saved.
            file_name = str(cb_params.cur_epoch_num) + "_" + str(cb_params.cur_step_num) + ".ckpt"
            # Save the network model.
            ms.save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Saved checkpoint, loss:{:8.7f}, current step num:{:4}.".format(cur_loss, cb_params.cur_step_num))

model.train(1, dataset_train, callbacks=[SaveCkptMonitor(5e-7)])
```

```text
    Saved checkpoint, loss:0.0000001, current step num: 253.
    Saved checkpoint, loss:0.0000005, current step num: 258.
    Saved checkpoint, loss:0.0000001, current step num: 265.
    Saved checkpoint, loss:0.0000000, current step num: 332.
    Saved checkpoint, loss:0.0000003, current step num: 358.
    Saved checkpoint, loss:0.0000003, current step num: 380.
    Saved checkpoint, loss:0.0000003, current step num: 395.
    Saved checkpoint, loss:0.0000005, current step num:1151.
    Saved checkpoint, loss:0.0000005, current step num:1358.
    Saved checkpoint, loss:0.0000002, current step num:1524.
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
