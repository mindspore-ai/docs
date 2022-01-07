# Initializer

`Ascend` `GPU` `CPU` `Model Development`

Translator: [Karlos Ma](https://gitee.com/Mavendetta985)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/initializer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

The Initializer class is the basic data structure used for initialization in MindSpore. Its subclasses contain several different types of data distribution (Zero, One, XavierUniform, Heuniform, Henormal, Constant, Uniform, Normal, TruncatedNormal). The following two parameter initialization modes, encapsulation operator and initializer method, are introduced in detail.

## Using the Initializer Method to Initialize Parameters

When initializer is used for parameter initialization, the parameters passed in are `init`, `shape`, `dtype`:
    -`init`: Supported subclasses of incoming `Tensor`, `STR`, `Subclass of Initializer`.
    -`shape`: Supported subclasses of incoming `list`, `tuple`, `int`.
    -`dtype`: Supported subclasses of incoming `mindspore.dtype`.

### The Parameter of Init is Tensor

The code sample is shown below:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import set_seed
from mindspore.common.initializer import initializer
import mindspore.ops as ops

set_seed(1)

input_data = Tensor(np.ones([16, 3, 10, 32, 32]), dtype=mstype.float32)
weight_init = Tensor(np.ones([32, 3, 4, 3, 3]), dtype=mstype.float32)
weight = initializer(weight_init, shape=[32, 3, 4, 3, 3])
conv3d = ops.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
output = conv3d(input_data, weight)
print(output)
```

The output is as follows:

```text
[[[[[108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]
    ...
    [108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]]
    ...
   [[108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]
    ...
    [108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]
    [108 108 108 ... 108 108 108]]]]]
```

### The Parameter of Init is Str

The code sample is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import set_seed
from mindspore.common.initializer import initializer
import mindspore.ops as ops

set_seed(1)

input_data = Tensor(np.ones([16, 3, 10, 32, 32]), dtype=mstype.float32)
weight = initializer('Normal', shape=[32, 3, 4, 3, 3], dtype=mstype.float32)
conv3d = ops.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
output = conv3d(input_data, weight)
print(output)
```

The output is as follows:

```text
[[[[[0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    ...
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    ...
   [[0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    ...
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]]]]
```

### The Parameter of Init is the Subclass of Initializer

The code sample is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import set_seed
import mindspore.ops as ops
from mindspore.common.initializer import Normal, initializer

set_seed(1)

input_data = Tensor(np.ones([16, 3, 10, 32, 32]), dtype=mstype.float32)
weight = initializer(Normal(0.2), shape=[32, 3, 4, 3, 3], dtype=mstype.float32)
conv3d = ops.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
output = conv3d(input_data, weight)
print(output)
```

The output is as follows:

```text
[[[[[0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    ...
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    ...
   [[0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    ...
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]]]]
```
