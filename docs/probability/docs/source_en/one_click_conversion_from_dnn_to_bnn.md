# One-click Conversion from DNN to BNN

<a href="https://gitee.com/mindspore/docs/blob/master/docs/probability/docs/source_en/one_click_conversion_from_dnn_to_bnn.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## One-click Conversion from DNN to BNN

For DNN researchers unfamiliar with the Bayesian model, MDP provides an advanced API `TransformToBNN` to convert the DNN model into the BNN model with one click. Currently, the API can be used in the LeNet, ResNet, MobileNet and VGG models. This example describes how to use the `TransformToBNN` API in the transforms module to convert DNNs into BNNs with one click. The overall process is as follows:

1. Define a DNN model.
2. Define the loss function and optimizer.
3. Function 1: Convert the entire model.
4. Function 2: Convert a layer of a specified type.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/master/tests/st/probability/transforms>.

### Defining the DNN Model

The deep neural network (DNN) model used in this example is LeNet5. After the definition is complete, print the name of its neural layer. Since the main convolutional layer and pooling layer are at the conversion level, this example also shows the conversion information of these two calculation layers in a targeted manner.

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    """Lenet network structure."""
    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

For the classic DNN model LeNet5 network convolutional layer has two layers conv1, conv2, fully connected layer is 3 layers: fc1, fc2, fc3.

### Defining the Loss Function and Optimizer

In this example, the loss function used is the cross-entropy loss function `nn.SoftmaxCrossEntropyWithLogits`, and the optimizer is the `Adam` function, which is `nn.AdamWeightDecay`.

Because the BNN of the entire model needs to be converted, it is necessary to associate the DNN network, loss function and optimizer into a complete calculation network, namely `train_network`.

```python
import pprint
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.probability import transforms
from mindspore import set_context, GRAPH_MODE

set_context(mode=GRAPH_MODE,device_target="GPU")

network = LeNet5()
lr = 0.01
momentum = 0.9
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = nn.AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
#optimizer = nn.Momentum(network.trainable_params(), lr, momentum)
net_with_loss = WithLossCell(network, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)

DNN_layer_name = [i.name for i in network.trainable_params()]
pprint.pprint(DNN_layer_name)
```

```text
['conv1.weight',
 'conv2.weight',
 'fc1.weight',
 'fc1.bias',
 'fc2.weight',
 'fc2.bias',
 'fc3.weight',
 'fc3.bias']
 ```

 The above-mentioned print information is the name of the convolutional layer and the fully connected layer that are not currently converted.

### Function 1: Converting the Entire Model

The `TransformToBNN` API in the `transforms` can convert the entire DNN model into a BNN model. The definition is as follows:

```python
bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
train_bnn_network = bnn_transformer.transform_to_bnn_model()

BNN_layer_name =[i.name for i in network.trainable_params()]
pprint.pprint(BNN_layer_name)
```

```text
['conv1.weight_posterior.mean',
 'conv1.weight_posterior.untransformed_std',
 'conv2.weight_posterior.mean',
 'conv2.weight_posterior.untransformed_std',
 'fc1.weight_posterior.mean',
 'fc1.weight_posterior.untransformed_std',
 'fc1.bias_posterior.mean',
 'fc1.bias_posterior.untransformed_std',
 'fc2.weight_posterior.mean',
 'fc2.weight_posterior.untransformed_std',
 'fc2.bias_posterior.mean',
 'fc2.bias_posterior.untransformed_std',
 'fc3.weight_posterior.mean',
 'fc3.weight_posterior.untransformed_std',
 'fc3.bias_posterior.mean',
 'fc3.bias_posterior.untransformed_std']
 ```

 The above-mentioned printed information is the convolutional layer and fully connected name after the overall conversion into a Bayesian network (BNN).

### Function 2: Converting a Layer of a Specified Type

The `transform_to_bnn_layer` method can convert a layer of a specified type (nn.Dense or nn.Conv2d) in the DNN model into a corresponding Bayesian layer. The definition is as follows:

```python
 transform_to_bnn_layer(dnn_layer, bnn_layer, get_args=None, add_args=None):
```

Parameter explanation:

- `dnn_layer`: Specify which type of DNN layer to convert into BNN layer.
- `bnn_layer`: Specify which type of BNN layer the DNN layer will be converted into.
- `get_args`: Specify which parameters to get from the DNN layer.
- `add_args`: Specify which parameters of the BNN layer to re-assign.

The code to convert the Dense layer in the DNN model into the corresponding Bayesian layer `DenseReparam` in MindSpore is as follows:

The `dnn_layer` parameter specifies a type of a DNN layer to be converted into a BNN layer, and the `bnn_layer` parameter specifies a type of a BNN layer to be converted into a DNN layer, `get_args` and `add_args` specify parameters obtained from the DNN layer and the parameters to be re-assigned to the BNN layer, respectively.

The code for converting a Dense layer in a DNN model into a corresponding Bayesian layer `DenseReparam` in MindSpore is as follows:

```python
from mindspore.nn.probability import bnn_layers

network = LeNet5(10)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = nn.AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
net_with_loss = WithLossCell(network, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)
bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
train_bnn_network = bnn_transformer.transform_to_bnn_layer(nn.Dense, bnn_layers.DenseReparam)
```

Execute the following code to view the structure of the converted network:

```python
DNN_layer_name = [i.name for i in network.trainable_params()]
pprint.pprint(DNN_layer_name)
```

```text
['conv1.weight',
 'conv2.weight',
 'fc1.weight_posterior.mean',
 'fc1.weight_posterior.untransformed_std',
 'fc1.bias_posterior.mean',
 'fc1.bias_posterior.untransformed_std',
 'fc2.weight_posterior.mean',
 'fc2.weight_posterior.untransformed_std',
 'fc2.bias_posterior.mean',
 'fc2.bias_posterior.untransformed_std',
 'fc3.weight_posterior.mean',
 'fc3.weight_posterior.untransformed_std',
 'fc3.bias_posterior.mean',
 'fc3.bias_posterior.untransformed_std']
```

As shown in the preceding information, the LeNet network transformed by `transform_to_bnn_layer` converts all the specified fully connected layers into Bayesian layers.
