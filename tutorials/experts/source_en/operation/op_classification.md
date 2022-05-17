# Operators Classification

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/operation/op_classification.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Operators are mainly divided into Primitivie operators and nn operators. For all operator support on Ascend AI processors, GPUs, and CPUs, see [Operator Support List](https://www.mindspore.cn/docs/en/master/note/operator_list.html).

## Primitive Operations

The primitive operators are the lowest-order operator interfaces open to users. A Primeive operator corresponds to a primitive, which encapsulates the specific implementation of the underlying Ascend, GPU, AICPU, CPU and other operators, providing users with basic operator capabilities.

The Primitive operator interface is the basis for building capabilities, such as higher-order interfaces, automatic differentiation, network models.

Primitive operators can be divided into [compute operators](#operator-related-to-compute) and [frame operators](#operators-related-to-frame). Compute operators are mainly responsible for specific calculations, while frame operators are mainly used for functions such as composition and automatic differentiation.

The composite interface provides some predefined combinatorial operators, such as clip_by_value operators, and functions involving graph transformations (GradOperation and Map). More composite interfaces can be found in the [composite interface](https://mindspore.cn/docs/en/master/api_python/mindspore.ops.html#composite).

The functional interface is provided to simplify the invocation process of the Primeive operator without attributes. The functional interface, composite interface, and Prime operator can all be imported from mindspore.ops.

For example, if you want to use the pow function and use the Primitive operator, you need to instantiate the Pow operator first, then you can directly use the tensor_pow of the functional interface to simplify the process, the code example is as follows:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
# Instantiate Pow operator before using Primitive operator
pow = ops.Pow()
output = pow(input_x, input_y)

# Directly use the functional interface
output = ops.tensor_pow(input_x, input_y)
print(output)
```

For more functional interfaces, see [functional interfaces](https://mindspore.cn/docs/en/master/api_python/mindspore.ops.functional.html).

### Operator Related to Compute

According to the function, the compute operators are mainly divided into neural network operators, mathematical operators, array operators, communication operators and so on.

#### Neural Network Operators

Neural network operators are mainly used to build network models, such as convolutional operator Conv2D, and maximum pooling operator MaxPool, see [Neural network operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html#neural-network-layer-operators).

The following code shows the use of maxPool, the maximum pooling operator:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.arange(1 * 3 * 3 * 4).reshape((1, 3, 3, 4)), mindspore.float32)
maxpool_op = ops.MaxPool(pad_mode="VALID", kernel_size=2, strides=1)
output = maxpool_op(x)
print(output)
```

#### Mathematical Operators

Mathematical operators are mainly operators developed for mathematical operations, such as additive operator Add, logarithmic operator Log, see [Mathematical Operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html#mathematical-operators).

The following code shows the use of logarithmic operator Log:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
log_oo = ops.Log()
output = log_oo(x)
print(output)
```

#### Array Operations

Array operators are mainly operators for array operations, such as sort operators Sort and transpose operators Transpose, and you can see the detailed in [Array Operator](https://mindspore.cn/docs/en/master/api_python/mindspore.ops.html#array-operation).

The following code shows the use of the transpose operators Transpose:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
input_perm = (0, 2, 1)
transpose_op = ops.Transpose()
output = transpose_op(input_x, input_perm)
print(output)
```

#### Communication Operators

Communication operators are mainly operators that communicate with each card during [multi-host training](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#multi-host-training), such as collecting operators AllGather, broadcast operators Widecast, etc., see [communication operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html#communication-operator).

The following code shows the use of the collecting operator AllGather:

```python
# This example should be run with two devices. Refer to the tutorial > Distributed Training on mindspore.cn
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.communication import init
from mindspore import Tensor, set_context, GRAPH_MODE

set_context(mode=GRAPH_MODE)
init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allgather = ops.AllGather()
    def construct(self, x):
        return self.allgather(x)

input_x = Tensor(np.ones([2, 8]).astype(np.float32))
net = Net()
output = net(input_x)
print(output)
```

The running results are as follows:

```text
[[1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]]
```

### Operators Related to Frame

`Mindspore.ops.composite` provides a number of combinatorial class operators involving graph transformations, such as `MultipeFuncGraph`, `HyperMap`, and `GradOperation`.

`MultitypeFuncGraph` is used to define a set of overloaded functions. The user can use this operator and use different implements according to different types. For detailed information, see [MultitypepeFuncGraph](https://www.mindspore.cn/tutorials/experts/en/master/operation/op_overload.html#multitypefuncgraph).

`HyperMap` can do specified operations on one or more sets of inputs and can be used with . For detailed information, see [HyperMap](https://www.mindspore.cn/tutorials/experts/en/master/operation/op_overload.html#hypermap).

`GradOperation` is used to generate gradients for input functions, which uses get_all, get_by_list, and sens_param parameters to control how to calculate the gradients. For detailed information, see [GradOperation](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) see .

## nn Operators

The nn operators are encapsulation of low-order APIs, mainly including convolutional layer operators, pooled layer operators, loss functions, optimizers.

The nn operators also provide part of the interfaces with the same name as the Primitive operator. The main role is to further encapsulate the Primitive operator, to provide users with a more friendly API. When the nn operator function meets the user's requirements, the user can directly use the nn operators, and when the nn operator function can not meet the user's specific requirements, you can use the low-level Primitive operator to achieve specific functions.

### Convolutional Layer Operators

Convolutional layer operators are mainly operators used in the convolutional layer of the model, such as convolutional operator Conv2d and transpose convolutional operator Conv2dTranspose. For the detailed information, see [Convolutional layer operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#convolutional-neural-network-layer).

The following code shows the use of convolutional operator Conv2d:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
output = net(x).shape
print(output)
```

### Pooling Layer Operators

Pooling layer operators are mainly operators used in the pooling layer of the model, such as average pooling operators AvgPool2d, maximum pooling operators MaxPool2d. For the detailed information, see [Pooling Layer Operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#pooling-layer).

The following code shows the use of the maximum pooling operators MaxPool2d:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

pool = nn.MaxPool2d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
output = pool(x)
print(output.shape)
```

### Loss Functions

The loss functions are mainly used to evaluate the degree of difference between the predicted value and the true value of the model, and the commonly used loss functions are BCEWithLogitsLoss and SoftmaxCrossEntropyWithLogits. For the detailed information, see [Loss Function](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#loss-function).

The following code shows the use of the SoftmaxCrossEntropyWithLogits loss function:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

loss = nn.SoftmaxCrossEntropyWithLogits()
logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
labels_np = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).astype(np.float32)
labels = Tensor(labels_np)
output = loss(logits, labels)
print(output)
```

### Optimizers

Optimizers are mainly used to calculate and update gradients, and commonly used optimizers are Adam and Momentum. For the detailed information, see [Optimizer](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#optimizer).

The following code shows the use of the Momentum optimizer:

```python
import mindspore.nn as nn
from mindspore import Model

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
# The conv_params's parameters will use a learning rate of default value 0.1 and a weight decay of 0.01 and
# grad centralization of True.
# The no_conv_params's parameters will use a learning rate of 0.01 and a weight decay of default value 0.0
# and grad centralization of False..
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
```