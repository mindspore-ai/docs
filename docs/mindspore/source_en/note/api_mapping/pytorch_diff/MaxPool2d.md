# Function Differences with torch.nn.MaxPool2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MaxPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.MaxPool2d

```text
class torch.nn.MaxPool2d(
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)(input) -> Tensor
```

For more information, see [torch.nn.MaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool2d.html).

## mindspore.nn.MaxPool2d

```text
class mindspore.nn.MaxPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.MaxPool2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MaxPool2d.html).

## Differences

PyTorch：Perform two-dimensional maximum pooling operations on the input multidimensional data.

MindSpore：The implementation function of API in MindSpore is basically the same as that of PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | kernel_size | kernel_size |Consistent function, no default values for PyTorch |
| | Parameter 2 | stride | stride |Consistent function, different default value |
| | Parameter 3 | padding | - | Implicit zero-padding added.When pad_mode='same', if the elements of padding are even, the elements of padding will be evenly distributed on the top and bottom of the feature map; while when the elements of padding are odd, PyTorch will preferentially padding on the left and top side of the input feature map, and MindSpore will preferentially padding on the right and bottom side of the feature map. For more details, see [Conv and Pooling](https://www.mindspore.cn/docs/en/master/migration_guide/typical_api_comparision.html#conv-and-pooling) |
| | Parameter 4 | dilation | - | Span length between elements in the window: the default value is 1, when the elements in the window are contiguous. If the value > 1, the elements in the window are spaced |
| | Parameter 5 | return_indices | - | Return index: If the value is True, the index of the corresponding element will be returned along with the maximum pooling result. Useful for subsequent calls to torch.nn.MaxUnpool2d|
| | Parameter 6 | ceil_mode | - | Control the output shape(N, C, L_{out}) in L_{out} to round up or down, and MindSpore defaults to round down |
| | Parameter 7 | input | x | Consistent function, different parameter names |
| | Parameter 8 | - | pad_mode | Control the padding mode, and PyTorch does not have this parameter |
| | Parameter 9 | - | data_format | The input data format can be "NHWC" or "NCHW". Default value: "NCHW" |

## Code Example 1

> Construct a pooling layer with a convolution kernel size of 1x3 and a step size of 1. The padding defaults to 0 and no element filling is performed. The default value of dilation is 1, and the elements in the window are contiguous. Pooling padding mode returns the output from a valid calculation without padding, and excess pixels that do not satisfy the calculation are discarded. With the same parameter settings, the two APIs achieve the same function to perform two-dimensional maximum pooling operations on the input multidimensional data.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

pool = torch.nn.MaxPool2d(kernel_size=3, stride=1)
x = tensor(np.random.randint(0, 10, [1, 2, 4, 4]), dtype=torch.float32)
output = pool(x)
result = output.shape
print(tuple(result))
# (1, 2, 2, 2)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
output = pool(x)
result = output.shape
print(result)
# (1, 2, 2, 2)
```

### Code Example 2

> When ceil_mode=True and pad_mode='same', both APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
x = torch.Tensor([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]])
print(x.dtype)
# torch.float32
output = max_pool(x)
print(output.numpy())
# [[[[ 3.  5.  7.  9. 10.]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

max_pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
x = Tensor([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]], mindspore.float32)
output = max_pool(x)
print(output)
# [[[[ 3.  5.  7.  9. 10.]]]]
```

### Code Example 3

> In PyTorch, when ceil_mode=False, set padding=1. In MindSpore, pad_mode='valid', first pad on the left and top side of x by ops.Pad(), then calculate the result of maximum pooling so that both APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

max_pool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
x = tensor(np.random.randint(0, 10, [1, 2, 4, 8]), dtype=torch.float32)
output = max_pool(x)
result = output.shape
print(tuple(result))
# (1, 2, 2, 4)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

max_pool = mindspore.nn.MaxPool2d(kernel_size=(3, 3), stride=2)
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 8]), mindspore.float32)
pad = ops.Pad(((0, 0), (0, 0), (1, 0), (1, 0)))
data = pad(Tensor(x))
output = max_pool(data)
print(output.shape)
# (1, 2, 2, 4)
```
