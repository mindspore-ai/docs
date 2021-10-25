# Operators Classification

`Ascend` `GPU` `CPU` `Beginner`

<!-- TOC -->

- [Operators Classification](#operators-classification)
    - [Overview](#overview)
    - [Tensor Operations](#tensor-operations)
        - [Mathematical Operators](#mathematical-operators)
            - [Scalar Operations](#scalar-operations)
            - [Vector Operations](#vector-operations)
            - [Matrix Operations](#matrix-operations)
        - [Broadcast Mechanism](#broadcast-mechanism)
    - [Network Operations](#network-operations)
        - [Feature Extraction](#feature-extraction)
        - [Activation Function](#activation-function)
        - [Loss Function](#loss-function)
        - [Optimization Algorithm](#optimization-algorithm)
    - [Array Operations](#array-operations)
        - [DType](#dtype)
        - [Cast](#cast)
        - [Shape](#shape)
    - [Image Operations](#image-operations)
    - [Encoding Operations](#encoding-operations)
        - [BoundingBoxEncode](#boundingboxencode)
        - [BoundingBoxDecode](#boundingboxdecode)
        - [IOU Computing](#iou-computing)
    - [Debugging Operations](#debugging-operations)
        - [HookBackward](#hookbackward)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/operators_classification.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Operators can be classified into some functional modules: tensor operations, network operations, array operations, image operations, encoding operations, debugging operations, and quantization operations. And they also involve some operator combinations related to graph transformation. For details about the supported operators on the Ascend AI processors, GPU, and CPU, see [Operator List](https://www.mindspore.cn/docs/note/en/master/operator_list.html).

## Tensor Operations

The tensor operations include the tensor structure operation and the tensor mathematical operation.

Tensor structure operations include tensor creation, index sharding, dimension transformation, and integration and splitting.

Tensor mathematical operations include scalar operations, vector operations, and matrix operations.

The following describes how to use the tensor mathematical operation and operation broadcast mechanism.

### Mathematical Operators

Tensor mathematical operators can be classified into scalar operator, vector operator, and matrix operator.

Scalar operators include addition, subtraction, multiplication, division, exponentiation, common functions such as trigonometric function, exponential function, and logarithmic function, and logical comparison operators.

#### Scalar Operations

Scalar operators are characterized by performing element-by-element operations on tensors.

Some scalar operators overload commonly used mathematical operators. In addition, the broadcast feature similar to NumPy is supported.

 The following code implements the exponentiation, where the base is input_x and the exponent is input_y:

```python
import numpy as np
import mindspore
from mindspore import Tensor

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
print(input_x**input_y)
```

 The following information is displayed:

```text
[ 1.  8. 64.]
```

##### Addition

The following code implements the addition of `input_x` and `input_y`:

```python
print(input_x + input_y)
```

 The following information is displayed:

```text
[4. 5. 7.]
```

##### Element-wise Multiplication

The following code implements the element-wise multiplication:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
mul = ops.Mul()
res = mul(input_x, input_y)

print(res)
```

 The following information is displayed:

```text
[4. 10. 18.]
```

##### Trigonometric Function

The following code implements Acos:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

acos = ops.ACos()
input_x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
output = acos(input_x)
print(output)
```

 The following information is displayed:

```text
[0.7377037 1.5307858 1.2661037 0.97641146]
```

#### Vector Operations

Vector operators perform operations on only one particular axis, mapping a vector to a scalar or another vector.

##### Squeeze

The following code implements the compression of a channel whose dimension of the third channel is 1:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
squeeze = ops.Squeeze(2)
output = squeeze(input_tensor)

print(output)
```

 The following information is displayed:

```text
[[1. 1.]
 [1. 1.]
 [1. 1.]]
```

#### Matrix Operations

Matrix operations include matrix multiplication, matrix norm, matrix determinant, matrix eigenvalue calculation, and matrix decomposition.

##### Matrix Multiplication

 The following code implements the matrix multiplication of input_x and input_y:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
input_y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
matmul = ops.MatMul()
output = matmul(input_x, input_y)

print(output)
```

The following information is displayed:

```text
[[3. 3. 3. 3.]]
```

### Broadcast Mechanism

Broadcast indicates that when the number of channels of each input variable is inconsistent, change the number of channels to obtain the result.

- The following code implements the broadcast mechanism:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

shape = (2, 3)
input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
broadcast_to = ops.BroadcastTo(shape)
output = broadcast_to(input_x)

print(output)
```

The following information is displayed:

```text
[[1. 2. 3.]
 [1. 2. 3.]]
```

## Network Operations

Network operations include feature extraction, activation function, loss function, and optimization algorithm.

### Feature Extraction

Feature extraction is a common operation in machine learning. The core of feature extraction is to extract more representative tensors than the original input.

Convolution Operation

The following code implements the 2D convolution operation which is one of the common convolution operations:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

input = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
conv2d = ops.Conv2D(out_channel=32, kernel_size=3)
res = conv2d(input, weight)

print(res)
```

The following information is displayed:

```text
[[[[288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   ...
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]]]

  ...

  [[288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   ...
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]]


 ...


  [[288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   ...
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]
   [288. 288. 288. ... 288. 288. 288.]]]]
```

Convolutional Backward Propagation Operator Operation

The following code implements the propagation operation of backward gradient operators. The outputs are stored in dout and weight:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

dout = Tensor(np.ones([10, 32, 30, 30]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
x = Tensor(np.ones([10, 32, 32, 32]))
conv2d_backprop_input = ops.Conv2DBackpropInput(out_channel=32, kernel_size=3)
res = conv2d_backprop_input(dout, weight, ops.shape(x))

print(res)
```

The following information is displayed:

```text
[[[[ 32.  64.  96. ...  96.  64.  32.]
   [ 64. 128. 192. ... 192. 128.  64.]
   [ 96. 192. 288. ... 288. 192.  96.]
   ...
   [ 96. 192. 288. ... 288. 192.  96.]
   [ 64. 128. 192. ... 192. 128.  64.]
   [ 32.  64.  96. ...  96.  64.  32.]]

  ...

  [[ 32.  64.  96. ...  96.  64.  32.]
   [ 64. 128. 192. ... 192. 128.  64.]
   [ 96. 192. 288. ... 288. 192.  96.]
   ...
   [ 96. 192. 288. ... 288. 192.  96.]
   [ 64. 128. 192. ... 192. 128.  64.]
   [ 32.  64.  96. ...  96.  64.  32.]]]]
```

### Activation Function

The following code implements the computation of the Softmax activation function:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
softmax = ops.Softmax()
res = softmax(input_x)

print(res)
```

The following information is displayed:

```text
[0.01165623 0.03168492 0.08612853 0.23412164 0.63640857]
```

### Loss Function

 L1Loss

 The following code implements the L1 loss function:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

loss = ops.SmoothL1Loss()
input_data = Tensor(np.array([1, 2, 3]), mindspore.float32)
target_data = Tensor(np.array([1, 2, 2]), mindspore.float32)
res = loss(input_data, target_data)
print(res)
```

 The following information is displayed:

```text
[0.  0.  0.5]
```

### Optimization Algorithm

 The following code implements the stochastic gradient descent (SGD) algorithm. The output is stored in result.

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

sgd = ops.SGD()
parameters = Tensor(np.array([2, -0.5, 1.7, 4]), mindspore.float32)
gradient = Tensor(np.array([1, -1, 0.5, 2]), mindspore.float32)
learning_rate = Tensor(0.01, mindspore.float32)
accum = Tensor(np.array([0.1, 0.3, -0.2, -0.1]), mindspore.float32)
momentum = Tensor(0.1, mindspore.float32)
stat = Tensor(np.array([1.5, -0.3, 0.2, -0.7]), mindspore.float32)
result = sgd(parameters, gradient, learning_rate, accum, momentum, stat)

print(result)
```

 The following information is displayed:

```text
(Tensor(shape=[4], dtype=Float32, value= [ 1.99000001e+00, -4.90300000e-01,  1.69500005e+00,  3.98009992e+00]),)
```

## Array Operations

Array operations refer to operations on arrays.

### DType

Returns a Tensor variable that has the same data type as the input and adapts to MindSpore. It is usually used in a MindSpore project.

The following is a code example:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
typea = ops.DType()(input_tensor)

print(typea)
```

 The following information is displayed:

```text
Float32
```

### Cast

Converts the input data type and outputs variables of the same type as the target data type.

The following is a code example:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = Tensor(input_np)
type_dst = mindspore.float16
cast = ops.Cast()
result = cast(input_x, type_dst)
print(result.dtype)
```

 The following information is displayed:

```text
Float16
```

### Shape

Returns the shape of the input data.

 The following code implements the operation of returning the input data input_tensor:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
shape = ops.Shape()
output = shape(input_tensor)
print(output)
```

 The following information is displayed:

```text
(3, 2, 1)
```

## Image Operations

The image operations include image preprocessing operations, for example, image cropping (for obtaining a large quantity of training samples) and resizing (for constructing an image pyramid).

 The following code implements the cropping and resizing operations:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

BATCH_SIZE = 1
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
image = np.random.normal(size=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]).astype(np.float32)
boxes = np.random.uniform(size=[NUM_BOXES, 4]).astype(np.float32)
box_index = np.random.uniform(size=[NUM_BOXES], low=0, high=BATCH_SIZE).astype(np.int32)
crop_size = (24, 24)
crop_and_resize = ops.CropAndResize()
output = crop_and_resize(Tensor(image), Tensor(boxes), Tensor(box_index), crop_size)
print(output.asnumpy())
```

The following information is displayed:

```text
[[[[ 6.51672244e-01 -1.85958534e-01 5.19907832e-01]
[ 1.53466597e-01 4.10562098e-01 6.26138210e-01]
[ 6.62892580e-01 3.81776541e-01 4.69261825e-01]
...
[-5.83377600e-01 -3.53377648e-02 -6.01786733e-01]
[ 1.36125124e+00 5.84172308e-02 -6.41442612e-02]
[-9.11651254e-01 -1.19495761e+00 1.96810793e-02]]

[[ 6.06956100e-03 -3.73778701e-01 1.88935513e-03]
[-1.06859171e+00 2.00272346e+00 1.37180305e+00]
[ 1.69524819e-01 2.90421434e-02 -4.12243098e-01]
...

[[-2.04489112e-01 2.36615837e-01 1.33802962e+00]
[ 1.08329034e+00 -9.00492966e-01 -8.21497202e-01]
[ 7.54147097e-02 -3.72897685e-01 -2.91040149e-02]
...
[ 1.12317121e+00 8.98950577e-01 4.22795087e-01]
[ 5.13781667e-01 5.12095273e-01 -3.68211865e-01]
[-7.04941899e-02 -1.09924078e+00 6.89047515e-01]]]]
```

> The preceding code runs on MindSpore of the Ascend version.

## Encoding Operations

The encoding operations include BoundingBox Encoding, BoundingBox Decoding, and IOU computing.

### BoundingBoxEncode

The box of the area where the object is located is encoded to obtain more concise information similar to PCA, facilitating subsequent tasks such as feature extraction, object detection, and image restoration.

The following code implements BoundingBox Encoding for anchor_box and groundtruth_box:

```python
from mindspore import Tensor
import mindspore.ops as ops
import mindspore

anchor_box = Tensor([[2,2,2,3],[2,2,2,3]],mindspore.float32)
groundtruth_box = Tensor([[1,2,1,4],[1,2,1,4]],mindspore.float32)
boundingbox_encode = ops.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
res = boundingbox_encode(anchor_box, groundtruth_box)
print(res)
```

 The following information is displayed:

```text
[[-1.          0.25        0.          0.40546513]
 [-1.          0.25        0.          0.40546513]]
```

### BoundingBoxDecode

After decoding the area location information, the encoder uses this operator to decode the information.

 Code implementation:

```python
from mindspore import Tensor
import mindspore.ops as ops
import mindspore

anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
deltas = Tensor([[3,1,2,2],[1,2,1,4]],mindspore.float32)
boundingbox_decode = ops.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), max_shape=(768, 1280), wh_ratio_clip=0.016)
res = boundingbox_decode(anchor_box, deltas)
print(res)
```

 The following information is displayed:

```text
[[ 4.194528   0.         0.         5.194528 ]
 [ 2.1408591  0.         3.8591409 60.59815  ]]
```

### IOU Computing

Computes the proportion of the intersection area and union area of the box where the predicted object is located and the box where the real object is located. It is often used as a loss function to optimize the model.

The following code implements the IOU computing between `anchor_boxes` and `gt_boxes`. The output is stored in out:

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
import mindspore

iou = ops.IOU()
anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
out = iou(anchor_boxes, gt_boxes)
print(out)
```

 The following information is displayed:

```text
[[ 0. -0.  0.]
 [ 0. -0.  0.]
 [ 0.  0.  0.]]
```

## Debugging Operations

The debugging operations refer to some common operators and operations used to debug a network, for example, HookBackward. These operations are very convenient and important for entry-level deep learning, greatly improving learning experience.

### HookBackward

Displays the gradient of intermediate variables. It is a common operator. Currently, only the PyNative mode is supported.

The following code implements the function of printing the gradient of the intermediate variable (x,y in this example):

```python
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
from mindspore import dtype as mstype
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE)

def hook_fn(grad_out):
    print(grad_out)

grad_all = ops.GradOperation(get_all=True)
hook = ops.HookBackward(hook_fn)

def hook_test(x, y):
    z = x * y
    z = hook(z)
    z = z * y
    return z

def backward(x, y):
    return grad_all(hook_test)(Tensor(x, mstype.float32), Tensor(y, mstype.float32))

print(backward(1, 2))
```

The following information is displayed:

```text
(Tensor(shape=[], dtype=Float32, value= 2),)
(Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
```
