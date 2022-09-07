# Training and Evaluation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/model/train_eval.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The preceding section describes the basic elements used by MindSpore to build a network, such as the basic network unit, loss function, optimizer, and evaluation function of MindSpore.

The following focuses on how to use these elements to customize training and evaluation networks.

## Building Training and Evaluation Processes

To build a training network, you need to build a feedforward network first, and then add the loss function, backward propagation, and optimizer on the basis of the feedforward network.

### Defining a Dataset

The following example defines the `get_data` function to generate sample data and corresponding labels, and defines the `create_dataset` function to load customized datasets.

```python
import mindspore.dataset as ds
import numpy as np

def get_data(num, w=2.0, b=3.0):
    """Generate sample data and corresponding labels."""
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16):
    """Generate a dataset."""
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset
```

### Building a Feedforward Network

Use `nn.Cell` to build a feedforward network. The following example defines a simple linear regression network `LinearNet`:

```python
import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)
```

### Building a Training Process

The `nn` module of MindSpore provides the training network encapsulation function `TrainOneStepCell` to encapsulate the network and optimizer. The parameters are as follows:

- `network`: specifies a training network. Only the single-output network is supported.
- `optimizer`: updates network parameters.
- `sens`: specifies the backward propagation input which is a scaling coefficient. The default value is 1.0.

The following example uses `nn.TrainOneStepCell` to encapsulate the defined linear regression network into a training network, perform training, and print the loss value.

In the sample code, `set_train` is used to specify whether the model is in training mode through the `mode` parameter. The default value of the `mode` parameter is True, indicating that the model is in training mode by default. If the value of `mode` is False, the model is in evaluation or inference mode.

```python
# Generate a training dataset.
train_dataset = create_dataset(num_data=160, batch_size=16)

net = LinearNet()
loss = nn.MSELoss()

# Connect the feedforward network and loss function.
net_with_loss = nn.WithLossCell(net, loss)
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# Define the training network and encapsulate the network and optimizer.
train_net = nn.TrainOneStepCell(net_with_loss, opt)
# Set the network to the training mode.
train_net.set_train()

# Training step process.
step = 0
epochs = 2
steps = train_dataset.get_dataset_size()

for epoch in range(epochs):
    for d in train_dataset.create_dict_iterator():
        result = train_net(d["data"], d["label"])
        print(f"Epoch: [{epoch} / {epochs}], "
              f"step: [{step} / {steps}], "
              f"loss: {result}")
        step = step + 1
```

```text
    Epoch: [0 / 2], step: [0 / 10], loss: 139.95065
    Epoch: [0 / 2], step: [1 / 10], loss: 77.1288
    Epoch: [0 / 2], step: [2 / 10], loss: 23.511435
    Epoch: [0 / 2], step: [3 / 10], loss: 15.275428
    Epoch: [0 / 2], step: [4 / 10], loss: 80.57905
    Epoch: [0 / 2], step: [5 / 10], loss: 86.396
    Epoch: [0 / 2], step: [6 / 10], loss: 78.92796
    Epoch: [0 / 2], step: [7 / 10], loss: 16.025606
    Epoch: [0 / 2], step: [8 / 10], loss: 2.996492
    Epoch: [0 / 2], step: [9 / 10], loss: 9.382026
    Epoch: [1 / 2], step: [10 / 10], loss: 46.85878
    Epoch: [1 / 2], step: [11 / 10], loss: 78.591515
    Epoch: [1 / 2], step: [12 / 10], loss: 39.523586
    Epoch: [1 / 2], step: [13 / 10], loss: 3.0048246
    Epoch: [1 / 2], step: [14 / 10], loss: 7.835808
    Epoch: [1 / 2], step: [15 / 10], loss: 27.37307
    Epoch: [1 / 2], step: [16 / 10], loss: 34.076313
    Epoch: [1 / 2], step: [17 / 10], loss: 54.53374
    Epoch: [1 / 2], step: [18 / 10], loss: 19.80341
    Epoch: [1 / 2], step: [19 / 10], loss: 1.8542566
```

### Building an Evaluation Process

The `nn` module of MindSpore provides the evaluation network encapsulation function `WithEvalCell` to evaluate the model training effect on the validation set. The parameters are as follows:

- `network`: specifies a feedforward network.
- `loss_fn`: specifies a loss function.
- `add_cast_fp32`: determines whether to adjust the data type to float32.

`nn.WithEvalCell` accepts only two inputs: `data` and its corresponding `label`. Use the previously defined feedforward network and loss function to build an evaluation network. The following is an example:

```python
eval_dataset = create_dataset(num_data=160, batch_size=16)

# Build an evaluation network.
eval_net = nn.WithEvalCell(net, loss)
eval_net.set_train(False)
loss = nn.Loss()
mae = nn.MAE()

mae.clear()
loss.clear()

# Evaluation process.
for data in eval_dataset.create_dict_iterator():
    outputs = eval_net(data["data"], data["label"])
    mae.update(outputs[1], outputs[2])
    loss.update(outputs[0])

# Evaluation result.
mae_result = mae.eval()
loss_result = loss.eval()

print("mae: ", mae_result)
print("loss: ", loss_result)
```

```text
    mae:  2.9597126245498657
    loss:  11.539738941192628
```

## Customized Training and Evaluation Networks

### Customized Training Network

[Customized Loss Functions](https://www.mindspore.cn/tutorials/en/master/advanced/modules/loss.html#customized-loss-functions) describes how to use `nn.WithLossCell` to connect the feedforward network to the loss function. This section describes how to customize the training network.

The following example defines the `CustomTrainOneStepCell` function to encapsulate the network and optimizer.

```python
import mindspore.ops as ops

class CustomTrainOneStepCell(nn.Cell):
    """Customize a training network.""

    def __init__(self, network, optimizer):
        """There are two input parameters: training network and optimizer."""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           #Define the feedforward network.
        self.network.set_grad()                          # Build a backward network.
        self.optimizer = optimizer                       #Define the optimizer.
        self.weights = self.optimizer.parameters         # Parameters to be updated.
        self.grad = ops.GradOperation(get_by_list=True)  #Obtain the gradient by backward propagation.

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # Compute the loss function value of the current input.
        grads = self.grad(self.network, self.weights)(*inputs)  # Perform backward propagation and compute the gradient.
        self.optimizer(grads)                                   # Use the optimizer to update weight parameters.
        return loss

net1 = LinearNet()   # Define the feedforward network.
loss = nn.MSELoss()  # Loss function


# Connect the feedforward network and loss function.
net_with_loss = nn.WithLossCell(net1, loss)
opt = nn.Momentum(net1.trainable_params(), learning_rate=0.005, momentum=0.9)

# Define the training network and encapsulate the network and optimizer.
train_net = CustomTrainOneStepCell(net_with_loss, opt)
# Set the network to the training mode.
train_net.set_train()

# Training step process.
step = 0
epochs = 2
steps = train_dataset.get_dataset_size()
for epoch in range(epochs):
    for d in train_dataset.create_dict_iterator():
        result = train_net(d["data"], d["label"])
        print(f"Epoch: [{epoch} / {epochs}], "
              f"step: [{step} / {steps}], "
              f"loss: {result}")
        step = step + 1
```

```text
    Epoch: [0 / 2], step: [0 / 10], loss: 70.774574
    Epoch: [0 / 2], step: [1 / 10], loss: 71.33737
    Epoch: [0 / 2], step: [2 / 10], loss: 63.126896
    Epoch: [0 / 2], step: [3 / 10], loss: 8.946123
    Epoch: [0 / 2], step: [4 / 10], loss: 32.131054
    Epoch: [0 / 2], step: [5 / 10], loss: 38.90644
    Epoch: [0 / 2], step: [6 / 10], loss: 126.410255
    Epoch: [0 / 2], step: [7 / 10], loss: 41.496185
    Epoch: [0 / 2], step: [8 / 10], loss: 5.7309575
    Epoch: [0 / 2], step: [9 / 10], loss: 16.104172
    Epoch: [1 / 2], step: [10 / 10], loss: 26.39038
    Epoch: [1 / 2], step: [11 / 10], loss: 52.73621
    Epoch: [1 / 2], step: [12 / 10], loss: 38.053413
    Epoch: [1 / 2], step: [13 / 10], loss: 4.555399
    Epoch: [1 / 2], step: [14 / 10], loss: 1.8704597
    Epoch: [1 / 2], step: [15 / 10], loss: 11.614007
    Epoch: [1 / 2], step: [16 / 10], loss: 25.868422
    Epoch: [1 / 2], step: [17 / 10], loss: 26.153322
    Epoch: [1 / 2], step: [18 / 10], loss: 9.847598
    Epoch: [1 / 2], step: [19 / 10], loss: 2.0711172
```

### Customized Evaluation Network

`nn.WithEvalCell` has only two inputs: `data` and `label`. Therefore, it is not applicable to scenarios with multiple data or labels. In this case, if you want to build an evaluation network, you need to customize the evaluation network.

If the loss function is not required as an evaluation metric, you do not need to define `loss_fn`. When multiple data or labels are input, you can refer to the following example to define the evaluation network.

```python
class CustomWithEvalCell(nn.Cell):

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label1, label2):
        """There are three pieces of input data: one piece of data and two corresponding labels."""
        outputs = self.network(data)
        return outputs, label1, label2

custom_eval_net = CustomWithEvalCell(net)
custom_eval_net.set_train(False)
```

```text
    CustomWithEvalCell<
      (network): LinearNet<
        (fc): Dense<input_channels=1, output_channels=1, has_bias=True>
        >
      >
```

## Sharing Network Weights

According to the preceding introduction, the forward network, training network, and evaluation network have different logic. Therefore, three networks are built when necessary. If a trained model is used for evaluation and inference, the weight values on the inference and evaluation network must be the same as those on the training network.

You can use the model saving and loading APIs to save the trained model and then load it to the evaluation and inference networks to ensure that the weight values are the same. Model training is completed on the training platform. However, model saving and loading are necessary for inference on the inference platform.

During network commissioning or model optimization using inference while training, the model training, evaluation, or inference is usually completed in the same Python script. In this case, the weight sharing mechanism of MindSpore ensures weight consistency between different networks.

When MindSpore is used to build different network structures, as long as these network structures are encapsulated based on an instance, the all weights in the instance are shared. If the weight in a network changes, the weights in other networks change synchronously.

In the following example, the weight sharing mechanism is used when the training and evaluation network is defined.

```python
# Instantiate the feedforward network.
net = LinearNet()
# Set the loss function and connect the feedforward network to the loss function.
loss = nn.MSELoss()
net_with_loss = nn.WithLossCell(net, loss)
# Set the optimizer.
opt = nn.Adam(params=net.trainable_params())

# Define a training network.
train_net = nn.TrainOneStepCell(net_with_loss, opt)
train_net.set_train()

# Build an evaluation network.
eval_net = nn.WithEvalCell(net, loss)
eval_net.set_train(False)
```

```text
    WithEvalCell<
      (_network): LinearNet<
        (fc): Dense<input_channels=1, output_channels=1, has_bias=True>
        >
      (_loss_fn): MSELoss<>
      >
```

Both `train_net` and `eval_net` are encapsulated based on the `net` instance. Therefore, you do not need to load the weight of `train_net` during model evaluation.

If the feedforward network is redefined when `eval_net` is built, `train_net` and `eval_net` do not share weights.

```python
# Define a training network.
train_net = nn.TrainOneStepCell(net_with_loss, opt)
train_net.set_train()

# Instantiate the feedforward network again.
net2 = LinearNet()
# Build an evaluation network.
eval_net = nn.WithEvalCell(net2, loss)
eval_net.set_train(False)
```

```text
    WithEvalCell<
      (_network): LinearNet<
        (fc): Dense<input_channels=1, output_channels=1, has_bias=True>
        >
      (_loss_fn): MSELoss<>
      >
```

In this case, to perform evaluation after model training, you need to load the weight in `train_net` to `eval_net`. When model training, evaluation, and inference are performed in the same script, it is easier to use the weight sharing mechanism.

In simple scenarios, the `Model` uses `nn.WithLossCell`, `nn.TrainOneStepCell`, and `nn.WithEvalCell` to build a training and evaluation network based on the feedforward network instance, the model itself ensures weight sharing among inference, training, and evaluation networks.

However, in the scenario where a customized model is used, you need to instantiate the forward network only once. If the feedforward network is instantiated separately when the training network and evaluation network are built, you need to manually load the weight of the training network when using eval to evaluate the model. Otherwise, the initial weight is used for model evaluation.
