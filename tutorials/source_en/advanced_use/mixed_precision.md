# Mixed Precision

`Ascend` `GPU` `Model Training` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r0.7/tutorials/source_en/advanced_use/mixed_precision.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. 
Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching 'reduce precision'.

## Computation Process

The following figure shows the typical computation process of mixed precision in MindSpore.

![mix precision](./images/mix_precision.jpg)

1. Parameters are stored in FP32 format.
2. During the forward computation, if an FP16 operator is involved, the operator input and parameters need to be cast from FP32 to FP16.
3. The loss layer is set to FP32.
4. During backward computation, the value is multiplied by Loss Scale to avoid underflow due to a small gradient.
5. The FP16 parameter is used for gradient computation, and the result is cast back to FP32.
6. Then, the value is divided by Loss scale to restore the multiplied gradient.
7. The optimizer checks whether the gradient overflows. If yes, the optimizer skips the update. If no, the optimizer uses FP32 to update the original parameters.

This document describes the computation process by using examples of automatic and manual mixed precision.

## Automatic Mixed Precision

To use the automatic mixed precision, you need to invoke the corresponding API, which takes the network to be trained and the optimizer as the input. This API converts the operators of the entire network into FP16 operators (except the `BatchNorm` and Loss operators).

The procedure is as follows:
1. Introduce the MindSpore mixed precision API.

2. Define the network. This step is the same as the common network definition. (You do not need to manually configure the precision of any specific operator.)

3. Use the `amp.build_train_network` API to encapsulate the network model and optimizer. In this step, MindSpore automatically converts the operators to the required format.

A code example is as follows:

```python
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.nn import Momentum
# The interface of Auto_mixed precision
from mindspore import amp

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")

# Define network
class Net(nn.Cell):
    def __init__(self, input_channel, out_channel):
        super(Net, self).__init__()
        self.dense = nn.Dense(input_channel, out_channel)
        self.relu = P.ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x


# Initialize network
net = Net(512, 128)

# Define training data, label
predict = Tensor(np.ones([64, 512]).astype(np.float32) * 0.01)
label = Tensor(np.zeros([64, 128]).astype(np.float32))

# Define Loss and Optimizer
loss = nn.SoftmaxCrossEntropyWithLogits()
optimizer = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
train_network = amp.build_train_network(net, optimizer, loss, level="O3", loss_scale_manager=None)

# Run training
output = train_network(predict, label)
```


## Manual Mixed Precision

MindSpore also supports manual mixed precision. It is assumed that only one dense layer in the network needs to be calculated by using FP32, and other layers are calculated by using FP16. The mixed precision is configured in the granularity of cell. The default format of a cell is FP32.

The following is the procedure for implementing manual mixed precision:
1. Define the network. This step is similar to step 2 in the automatic mixed precision. 

2. Configure the mixed precision. Use `net.to_float(mstype.float16)` to set all operators of the cell and its sub-cells to FP16. Then, configure the dense to FP32.

3. Use TrainOneStepCell to encapsulate the network model and optimizer.

A code example is as follows:

```python
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn import Momentum

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")

# Define network
class Net(nn.Cell):
    def __init__(self, input_channel, out_channel):
        super(Net, self).__init__()
        self.dense = nn.Dense(input_channel, out_channel)
        self.relu = P.ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x

# Initialize network and set mixing precision
net = Net(512, 128)
net.to_float(mstype.float16)
net.dense.to_float(mstype.float32)

# Define training data, label
predict = Tensor(np.ones([64, 512]).astype(np.float32) * 0.01)
label = Tensor(np.zeros([64, 128]).astype(np.float32))

# Define Loss and Optimizer
loss = nn.SoftmaxCrossEntropyWithLogits()
optimizer = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
net_with_loss = WithLossCell(net, loss)
train_network = TrainOneStepCell(net_with_loss, optimizer)
train_network.set_train()

# Run training
output = train_network(predict, label)
```
