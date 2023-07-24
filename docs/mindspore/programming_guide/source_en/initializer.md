# Initialization of Network Parameters

Translator: [Karlos Ma](https://gitee.com/Mavendetta985)

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/initializer.md)

## Overview

MindSpore provides a weight initialization module, which allows users to initialize network parameters by encapsulated operators and initializer methods to call strings, initializer subclasses, or custom Tensors. The Initializer class is the basic data structure used for initialization in MindSpore. Its subclasses contain several different types of data distribution (Zero, One, XavierUniform, Heuniform, Henormal, Constant, Uniform, Normal, TruncatedNormal). The following two parameter initialization modes, encapsulation operator and initializer method, are introduced in detail.

## Using Encapsulation Operator to Initialize Parameters

Mindspore provides a variety of methods of initializing parameters, and encapsulates parameter initialization functions in some operators. This section will introduce the method of initialization of parameters by operators with parameter initialization function. Taking `Conv2D` operator as an example, it will introduce the initialization of parameters in the network by strings, `Initializer` subclass and custom `Tensor`, etc. `Normal`, a subclass of `Initializer`, is used in the following code examples and can be replaced with any of the subclasses of Initializer in the code examples.

### Character String

Network parameters are initialized using a string. The contents of the string need to be consistent with the name of the `Initializer` subclass. Initialization using a string will use the default parameters in the `Initializer` subclass. For example, using the string `Normal` is equivalent to using the `Initializer` subclass `Normal()`. The code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import set_seed

set_seed(1)

input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init='Normal')
output = net(input_data)
print(output)
```

```text
[[[[ 3.10382620e-02  4.38603461e-02  4.38603461e-02 ...  4.38603461e-02
     4.38603461e-02  1.38719045e-02]
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   ...
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   [ 9.66199022e-03  1.24104535e-02  1.24104535e-02 ...  1.24104535e-02
     1.24104535e-02 -1.38977719e-02]]

  ...

  [[ 3.98553275e-02 -1.35465711e-03 -1.35465711e-03 ... -1.35465711e-03
    -1.35465711e-03 -1.00310734e-02]
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   ...
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   [ 1.33139016e-02  6.74417242e-05  6.74417242e-05 ...  6.74417242e-05
     6.74417242e-05 -2.27325838e-02]]]]
```

### Initializer Subclass

`Initializer` subclass is used to initialize network parameters, which is similar to the effect of using string to initialize parameters.  The difference is that using string to initialize parameters uses the default parameter of the `Initializer` subclass. If you want to use the parameters in the `Initializer` subclass, the `Initializer` subclass must be used to initialize the parameters. Taking `Normal(0.2)` as an example, the code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.common.initializer import Normal

set_seed(1)

input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init=Normal(0.2))
output = net(input_data)
print(output)
```

```text
[[[[ 6.2076533e-01  8.7720710e-01  8.7720710e-01 ...  8.7720710e-01
     8.7720710e-01  2.7743810e-01]
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   ...
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   [ 1.9323981e-01  2.4820906e-01  2.4820906e-01 ...  2.4820906e-01
     2.4820906e-01 -2.7795550e-01]]

  ...

  [[ 7.9710668e-01 -2.7093157e-02 -2.7093157e-02 ... -2.7093157e-02
    -2.7093157e-02 -2.0062150e-01]
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   ...
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   [ 2.6627803e-01  1.3488382e-03  1.3488382e-03 ...  1.3488382e-03
     1.3488382e-03 -4.5465171e-01]]]]
```

### The Custom of the Tensor

In addition to the above two initialization methods,  when the network wants to use data types that are not available in MindSpore, users can customize `Tensor` to initialize the parameters. The code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype

weight = Tensor(np.ones([64, 3, 3, 3]), dtype=mstype.float32)
input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init=weight)
output = net(input_data)
print(output)
```

```text
[[[[12. 18. 18. ... 18. 18. 12.]
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   ...
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   [12. 18. 18. ... 18. 18. 12.]]

  ...

  [[12. 18. 18. ... 18. 18. 12.]
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   ...
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   [12. 18. 18. ... 18. 18. 12.]]]]
```

## Using the Initializer Method to Initialize Parameters

In the above code sample, the method of Parameter initialization in the network is given. For example, NN layer is used to encapsulate a `Conv2D` operator in the network, and the Parameter `weight_init` is passed into a `Conv2D` operator as the data type to be initialized. The operator will be initialized by calling `Parameter` class. Then the `initializer` method encapsulated in the `Parameter` class is called to initialize the parameters. However, some operators do not encapsulate the function of parameter initialization internally like `Conv2D`. For example, the weights of `Conv3D` operators are passed to `Conv3D` operators as parameters. In this case, it is necessary to manually define the initialization of weights.

When initializing a parameter, you can use the `Initializer` method to initialize the parameter by calling different data types in the `Initializer` subclasses, resulting in different types of data.

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
from mindspore.common import set_seed
from mindspore.common.initializer import initializer
from mindspore.ops.operations import nn_ops as nps

set_seed(1)

input_data = Tensor(np.ones([16, 3, 10, 32, 32]), dtype=mstype.float32)
weight_init = Tensor(np.ones([32, 3, 4, 3, 3]), dtype=mstype.float32)
weight = initializer(weight_init, shape=[32, 3, 4, 3, 3])
conv3d = nps.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
output = conv3d(input_data, weight)
print(output)
```

```text
The output is as follows:
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
from mindspore.common import set_seed
from mindspore.common.initializer import initializer
from mindspore.ops.operations import nn_ops as nps

set_seed(1)

input_data = Tensor(np.ones([16, 3, 10, 32, 32]), dtype=mstype.float32)
weight = initializer('Normal', shape=[32, 3, 4, 3, 3], dtype=mstype.float32)
conv3d = nps.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
output = conv3d(input_data, weight)
print(output)
```

```text
The output is as follows:
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
from mindspore.common import set_seed
from mindspore.ops.operations import nn_ops as nps
from mindspore.common.initializer import Normal, initializer

set_seed(1)

input_data = Tensor(np.ones([16, 3, 10, 32, 32]), dtype=mstype.float32)
weight = initializer(Normal(0.2), shape=[32, 3, 4, 3, 3], dtype=mstype.float32)
conv3d = nps.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
output = conv3d(input_data, weight)
print(output)
```

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

### Application in Parameter

The code sample is as follows:

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.ops import operations as ops
from mindspore import Tensor, Parameter, context
from mindspore.common.initializer import Normal, initializer

set_seed(1)

weight1 = Parameter(initializer('Normal', [5, 4], mstype.float32), name="w1")
weight2 = Parameter(initializer(Normal(0.2), [5, 4], mstype.float32), name="w2")
input_data = Tensor(np.arange(20).reshape(5, 4), dtype=mstype.float32)
net = ops.Add()
output = net(input_data, weight1)
output = net(output, weight2)
print(output)
```

```text
[[-0.3305102  1.0412874  2.0412874  3.0412874]
 [ 4.0412874  4.9479127  5.9479127  6.9479127]
 [ 7.947912   9.063009  10.063009  11.063009 ]
 [12.063009  13.536987  14.536987  14.857441 ]
 [15.751231  17.073082  17.808317  19.364822 ]]
```
