# Case: Linear Fitting

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/train/linear_fitting.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

MindSpore provides high-level, medium-level, and low-level APIs. For details, see [API Level Structure](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html#api-level-structure).

To facilitate the control of the network execution process, MindSpore provides the high-level training and inference API `mindspore.Model`. By specifying the neural network model to be trained and common training settings, MindSpore calls the `train` and `eval` methods to train and infer the network. In addition, if you want to customize a specific module, you can call the corresponding medium- and low-level APIs to define the network training process.

The following uses the medium- and low-level APIs provided by MindSpore to fit linear functions.

$$f(x) = 2x + 3  \tag {1}$$

Before initializing the network, you need to configure the `context` parameter to control the program execution policy. For example, configure the static graph or dynamic graph mode and configure the hardware environment for network running.

The following describes how to configure information and use medium- and low-level APIs provided by MindSpore to customize loss functions, optimizers, training processes, metrics, and evaluation processes.

## Configuration Information

Before initializing the network, you need to configure the `context` parameter to control the program execution policy. For example, configure the static graph or dynamic graph mode and configure the hardware environment for network running. Before initializing the network, you need to configure the `context` parameter to control the program execution policy. The following describes the execution mode management and hardware management.

### Execution Mode

MindSpore supports two running modes: Graph and PyNative. By default, MindSpore uses the Graph mode, and the PyNative mode is used for debugging.

- Graph mode (static graph mode): The neural network model is built into an entire graph and then delivered to the hardware for execution. This mode uses graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.

- PyNative mode (dynamic graph mode): Operators in the neural network are delivered to the hardware one by one for execution. This mode facilitates code writing and neural network model debugging.

MindSpore provides a unified encoding mode for static and dynamic graphs, significantly enhancing compatibility between both types of graphs. This enables you to switch between the static and dynamic graph modes by changing only one line of code, eliminating the need to develop multiple sets of code. When switching the mode, pay attention to the [constraints](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) of the target mode.

Set the running mode to dynamic graph mode.

```python
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)
```

Similarly, when MindSpore is in dynamic image mode, you can run the `set_context(mode=GRAPH_MODE)` command to switch to the static graph mode.

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)
```

### Hardware Management

Hardware management involves the `device_target` and `device_id` parameters.

- `device_target`: target device to be run. The value can be `Ascend`, `GPU`, or `CPU`. You can set this parameter based on the actual situation or use the default value.

- `device_id`: ID of the target device. The value is in the range of [0,`device_num_per_host` - 1]. `device_num_per_host` indicates the total number of devices on the server. The value of `device_num_per_host` cannot exceed 4096. The default value of `device_id` is 0.

> When the program is executed in non-distributed mode, you can set `device_id` to determine the ID of the device where the program is executed to avoid device usage conflicts.

A code example is as follows:

```Python
import mindspore as mst

ms.set_context(device_target="Ascend", device_id=6)
```

## Processing Datasets

### Generating a Dataset

Define the dataset generation function `get_data` to generate the training dataset and test dataset.

Since linear data is fitted, the required training datasets should be randomly distributed around the objective function. Assume that the objective function to be fitted is $f(x)=2x+3$. $f(x)=2x+3+noise$ is used to generate training datasets, and `noise` is a random value that complies with standard normal distribution rules.

```python
import numpy as np

def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)
```

Use get_data to generate 50 groups of evaluation data and visualize the data.

```python
import matplotlib.pyplot as plt

train_data = list(get_data(50))
x_target_label = np.array([-10, 10, 0.1])
y_target_label = x_target_label * 2 + 3
x_eval_label, y_eval_label = zip(*train_data)

plt.scatter(x_eval_label, y_eval_label, color="red", s=5)
plt.plot(x_target_label, y_target_label, color="green")
plt.title("Eval data")
plt.show()
```

![png](output_8_0.png)

In the figure shown above, the green line indicates the target function, and the red point indicates the evaluation data (`train_data`).

### Loading a Dataset

Loads the dataset generated by the `get_data` function to the system memory and performs basic data processing operations.

- `ds.GeneratorDataset`: converts the generated data into a MindSpore dataset and saves the x and y values of the generated data to arrays of `data` and `label`.
- `batch`: combines `batch_size` pieces of data into a batch.
- `repeat`: Multiplies the number of data sets.

```python
from mindspore import dataset as ds

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data
```

Use the dataset augmentation function to generate training data. Use the defined `create_dataset` to augment the generated 1600 pieces of data into 100 datasets with the shape of 16 x 1.

```python
data_number = 1600
batch_number = 16
repeat_number = 1

ds_train = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)
print("The dataset size of ds_train:", ds_train.get_dataset_size())
step_size = ds_train.get_dataset_size()
dict_datasets = next(ds_train.create_dict_iterator())

print(dict_datasets.keys())
print("The x label value shape:", dict_datasets["data"].shape)
print("The y label value shape:", dict_datasets["label"].shape)
```

```text
    The dataset size of ds_train: 100
    dict_keys(['data', 'label'])
    The x label value shape: (16, 1)
    The y label value shape: (16, 1)
```

## Defining a Network Model

The `mindspore.nn` class is the base class for setting up all networks and the basic unit of a network. In order to customize a network, you can inherit the `nn.Cell` class and overwrite the `__init__` and `construct` methods.

The `mindspore.ops` module provides the implementation of basic operators. The `nn.Cell` module further encapsulates basic operators. You can flexibly use different operators as required.

The following example uses `nn.Cell` to build a simple fully-connected network for subsequent customized content. In MindSpore, use `nn.Dense` to generate a linear function model with a single data input and a single data output.

$$f(x)=wx+b\tag{2}$$

Use the Normal operator to randomly initialize the $w$ and $b$ parameters in formula (2).

```python
from mindspore import nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        fx = self.fc(x)
        return fx
```

After initializing the network model, visualize the initialized network function and training dataset to understand the model function before fitting.

```python
import mindspore as ms

net = LinearNet()  # Initialize the linear regression network.

model_params = net.trainable_params()  # Obtain network parameters w and b before training.

x_model_label = np.array([-10, 10, 0.1])
y_model_label = (x_model_label * model_params[0].asnumpy()[0] + model_params[1].asnumpy()[0])

plt.axis([-10, 10, -20, 25])
plt.scatter(x_eval_label, y_eval_label, color="red", s=5)
plt.plot(x_model_label, y_model_label, color="blue")
plt.plot(x_target_label, y_target_label, color="green")
plt.show()
```

![png](output_16_0.png)

## Customized Loss Functions

A loss function is used to measure the difference between the predicted value and the actual value. In deep learning, model training is a process of reducing a loss function value through continuous iteration. Therefore, it is very important to select a loss function in a model training process. Defining a good loss function can help the loss function value converge faster and achieve better accuracy.

[mindspore.nn](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#loss-function) provides many common loss functions for users to select and allows users to customize loss functions as required.

When customizing a loss function class, you can inherit the base class `nn.Cell` of the network or the base class `nn.LossBase` of the loss function. `nn.LossBase` is based on `nn.Cell` and provides the `get_loss` method. The `reduction` parameter is used to obtain a sum or mean loss value and output a scalar. The following describes how to define the mean absolute error (MAE) function by inheriting LossBase. The formula of the MAE algorithm is as follows:

$$ loss= \frac{1}{m}\sum_{i=1}^m\lvert y_i-f(x_i) \rvert \tag{3}$$

In the preceding formula, $f(x)$ indicates the predicted value, $y$ indicates the actual value of the sample, and $loss$ indicates the mean distance between the predicted value and the actual value.

When using the method inherited from LossBase to define the loss function, you need to rewrite the `__init__` and `construct` methods and use the `get_loss` method to compute the loss. The sample code is as follows:

```python
from mindspore import nn, ops

class MyMAELoss(nn.LossBase):
    """Define the loss."""
    def __init__(self):
        super(MyMAELoss, self).__init__()
        self.abs = ops.Abs()

    def construct(self, predict, target):
        x = self.abs(target - predict)
        return self.get_loss(x)
```

## Customized Optimizer

During model training, the optimizer is used to compute and update network parameters. A proper optimizer can effectively reduce the training time and improve model performance.

[mindspore.nn](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#optimizer) provides many common optimizers for users to select and allows users to customize optimizers as required.

When customizing an optimizer, you can inherit the optimizer base class `nn.Optimizer` and rewrite the `__init__` and `construct` methods to update parameters.

The following example implements the customized optimizer Momentum (SGD algorithm with momentum):

$$ v_{t+1} = v_t × u+grad \tag{4}$$

$$p_{t+1} = p_t - lr × v_{t+1} \tag{5}$$

$grad$, $lr$, $p$, $v$, and $u$ respectively represent a gradient, a learning rate, a weight parameter, a momentum parameter, and an initial speed.

```python
import mindspore as ms
from mindspore import nn, ops

class MyMomentum(nn.Optimizer):
    """Define the optimizer."""

    def __init__(self, params, learning_rate, momentum=0.9):
        super(MyMomentum, self).__init__(learning_rate, params)
        self.moment = ms.Parameter(ms.Tensor(momentum, ms.float32), name="moment")
        self.momentum = self.parameters.clone(prefix="momentum", init="zeros")
        self.assign = ops.Assign()

    def construct(self, gradients):
        """The input of construct is gradient. Gradients are automatically transferred during training."""
        lr = self.get_lr()
        params = self.parameters  # Weight parameter to be updated
        for i in range(len(params)):
            self.assign(self.momentum[i], self.momentum[i] * self.moment + gradients[i])
            update = params[i] - self.momentum[i] * lr  # SGD algorithm with momentum
            self.assign(params[i], update)
        return params
```

## Customized Training Process

`mindspore.Model` provides `train` and `eval` APIs for users to use during training. However, this API does not apply to all scenarios, such as multi-data and multi-label scenarios, where users need to define the training process.

The following uses linear regression as an example to describe the customized training process. First, define the loss network and connect the forward network to the loss function. Then, define the training process. Generally, the training process inherits `nn.TrainOneStepCell`. `nn.TrainOneStepCell` encapsulates the loss network and optimizer to implement the backward propagation network to update the weight parameters.

### Defining a Loss Network

Define the loss network `MyWithLossCell` to connect the feedforward network to the loss function.

```python
class MyWithLossCell(nn.Cell):
    """Define the loss network."""

    def __init__(self, backbone, loss_fn):
        """Transfer the feedforward network and loss function as parameters during instantiation."""
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        """Connect the feedforward network and loss function."""
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        """Backbone network to be encapsulated."""
        return self.backbone
```

### Defining the Training Process

Define the training process `MyTrainStep`. This class inherits `nn.TrainOneStepCell`. `nn.TrainOneStepCell` encapsulates the loss network and optimizer. During training, the `ops.GradOperation` operator is used to obtain the gradient, the optimizer is used to update the weight.

```python
class MyTrainStep(nn.TrainOneStepCell):
    """Define the training process."""

    def __init__(self, network, optimizer):
        """Initialize parameters."""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """Build the training process."""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)
```

The following defines the drawing function `plot_model_and_datasets` to draw the test data, objective function, and network model fitting function, and view the loss value.

```python
from IPython import display
import matplotlib.pyplot as plt
import time

def plot_model_and_datasets(net, data, loss):
    weight = net.trainable_params()[0]
    bias = net.trainable_params()[1]
    x = np.arange(-10, 10, 0.1)
    y = x * ms.Tensor(weight).asnumpy()[0][0] + ms.Tensor(bias).asnumpy()[0]
    x1, y1 = zip(*data)
    x_target = x
    y_target = x_target * 2 + 3

    plt.axis([-11, 11, -20, 25])
    plt.scatter(x1, y1, color="red", s=5)        # Raw data
    plt.plot(x, y, color="blue")                 # Predicted data
    plt.plot(x_target, y_target, color="green")  # Fitting function
    plt.title(f"Loss:{loss}")                    # Printed loss value

    plt.show()
    time.sleep(0.2)
    display.clear_output(wait=True)
```

### Training

Use the training data `ds_train` to train the training network `train_net` and visualize the training process.

```python
loss_func = MyMAELoss ()                         # Loss function
opt = MyMomentum(net.trainable_params(), 0.01)  # Optimizer

net_with_criterion = MyWithLossCell(net, loss_func)  # Build a loss network.
train_net = MyTrainStep(net_with_criterion, opt)     # Build a training network.

for data in ds_train.create_dict_iterator():
    train_net(data['data'], data['label'])                  # Perform training and update the weight.
    loss = net_with_criterion(data['data'], data['label'])  # Compute the loss value.
    plot_model_and_datasets(train_net, train_data, loss)    # Visualize the.
```

![png](output_28_0.png)

## Customized Evaluation Metrics

When a training task is complete, an evaluation function (Metric) is often required to evaluate the quality of a model. Common evaluation metrics include confusion matrix, accuracy, precision, and recall.

The [mindspore.nn](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#evaluation-metrics) module provides common evaluation functions. You can also define evaluation metrics as required. The customized Metric function needs to inherit the `nn.Metric` parent class and re-implement the `clear`, `update`, and `eval` methods in the parent class. The following formula shows the mean absolute error (MAE) algorithm. The following uses MAE as an example to describe the three functions and their usage.

$$ MAE=\frac{1}{n}\sum_{i=1}^n\lvert y\_pred_i - y_i \rvert \tag{6}$$

- `clear`: initializes related internal parameters.
- `update`: receives network prediction output and labels, computes errors, and updates internal evaluation results. Generally, the calculation is performed after each step and the statistical value is updated.
- `eval`: computes the final evaluation result after each epoch ends.

```python
class MyMAE(nn.Metric):
    """Define metrics."""

    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """Initialize variables abs_error_sum and samples_num."""
        self.abs_error_sum = 0
        self.samples_num = 0

    def update(self, *inputs):
        """Update abs_error_sum and samples_num."""
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        # Compute the absolute error between the predicted value and the actual value.
        error_abs = np.abs(y.reshape(y_pred.shape) - y_pred)
        self.abs_error_sum += error_abs.sum()
        self.samples_num += y.shape[0]  # Total number of samples

    def eval(self):
        """Compute the final evaluation result.""
        return self.abs_error_sum / self.samples_num
```

## Customized Evaluation Process

The mindspore.nn module provides the evaluation network packaging function [nn.WithEvalCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.WithEvalCell.html#mindspore.nn.WithEvalCell). Because `nn.WithEvalCell` has only two inputs `data` and `label`, it is not applicable to the scenario with multiple data or labels. Therefore, you need to customize the evaluation network. For details about how to customize the evaluation network in the multi-label scenario, see [Customized Training and Evaluation Networks](https://www.mindspore.cn/tutorials/en/master/advanced/train/train_eval.html).

The following example implements a simple customized evaluation network `MyWithEvalCell`. Enter the input `data` and `label`.

```python
class MyWithEvalCell(nn.Cell):
    """Define the evaluation process."""

    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        outputs = self.network(data)
        return outputs, label
```

Perform inference and evaluation:

```python
data_number = 160
batch_number = 16
repeat_number = 1

# Obtain evaluation data.
ds_eval = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)

eval_net = MyWithEvalCell(net)  # Define the evaluation network.
eval_net.set_train(False)
mae = MyMAE()

# Execute the inference process.
for data in ds_eval.create_dict_iterator():
    output, eval_y = eval_net(data['data'], data['label'])
    mae.update(output, eval_y)

mae_result = mae.eval()
print("MAE: ", mae_result)
```

```text
    MAE:  0.9605088472366333
```

Output evaluation error. The effect of MAE is similar to that of the model in the training set.

## Saving and Exporting a Model

Save the trained model parameters to a checkpoint (CKPT) file, and export the checkpoint file as a MindIR file for cross-platform inference.

```python
import numpy as np
import mindspore as ms

ms.save_checkpoint(net, "./linear.ckpt")          # Save model parameters in a CKPT file.
param_dict = ms.load_checkpoint("./linear.ckpt")  # Save the model parameters to the param_dict dictionary.

# View the model parameters.
for param in param_dict:
    print(param, ":", param_dict[param].asnumpy())

net1 = LinearNet()
input_np = np.random.uniform(0.0, 1.0, size=[1, 1]).astype(np.float32)
ms.export(net1, ms.Tensor(input_np), file_name='linear', file_format='MINDIR')
```

```text
    fc.weight : [[1.894384]]
    fc.bias : [3.0015702]
```
