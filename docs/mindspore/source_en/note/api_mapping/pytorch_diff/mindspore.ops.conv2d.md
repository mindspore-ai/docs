# Function Differences with torch.nn.functional.conv2d

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.ops.conv2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.conv2d

```text
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

For more information, see [torch.nn.functional.conv2d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.conv2d).

## mindspore.ops.conv2d

```text
mindspore.ops.conv2d(inputs, weight, pad_mode="valid", padding=0, stride=1, dilation=1, group=1)
```

For more information, see [mindspore.ops.conv2d](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.conv2d.html).

## Differences

PyTorch: Compute a two-dimensional convolution on the input Tensor, pass in manually the parameters of the convolution kernel (i.e., the weight matrix). This weight matrix can be a pre-trained model parameter or a manually-set matrix.

MindSpore: It is basically the same as the functions implemented by PyTorch, but there are bias differences and filling differences.

1. Bias difference: MindSpore does not have bias parameters.
2. Fill difference: MindSpore fills the input by default, while PyTorch does not fill by default. At the same time, MindSpore filling mode options and behavior are different from that of PyTorch. The specific differences in filling behavior are as follows.

### Filling Behavior Difference

1. Options of PyTorch padding are int, tuples of ints, and the default is 0. The padding parameter is used to control the amount and position of padding. For conv2d, when padding is specified as int, a padding number of paddings will be performed on the top and bottom of the input (if the default value is 0, it means no padding). when padding is specified as a tuple, a specified number of times of padding will be performed on top and bottom and left and right according to the input of the tuple.

2. Options of MindSpore pad_mode are 'same', 'valid', 'pad', and the parameter padding can only be inputted as int or tuple of ints. The detailed meaning of the padding parameter is as follows.

    When "pad_mode" is set to 'pad', "MindSpore" can set the "padding" parameter to a positive integer greater than or equal to 0. Zero filling will be carried out "padding" times around the input(if it is the default value of 0, it will not fill). If padding is a tuple consisted of 4 int, the top, bottom, left and right padding are equal to padding[0], padding[1], padding[2] and padding[3] respectively. When "pad_mode" is the other two modes, the "padding" parameter must be set to 0 only. When "pad_mode" is set to 'valid' mode, it will not fill, and the convolution will only be carried out within the range of the feature map. If "pad_mode" is set to 'same' mode, when the padding element is an even number, padding elements are evenly distributed on the top, bottom, left, and right of the feature map. If the number of elements requiring "padding" is odd, "MindSpore" will preferentially fill the right and lower sides of the feature map (different from PyTorch, similar to TensorFlow).

3. Parameter name differences: PyTorch and MindSpore have some differences in their parameter names, as follows:

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | inputs |- |
| | Parameter 2 | weight | weight |- |
| | Parameter 3 | bias | - | Bias parameter is missing in MindSpore |
| | Parameter 4 | stride | stride |- |
| | Parameter 5 | padding | padding |Refer to the above for specific differences|
| | Parameter 6 | dilation | dilation |-|
| | Parameter 7 | groups | group |Same function, different parameter names|
| | Parameter 9 | - | pad_mode |Refer to the above for specific differences|

### Code Example

> If PyTorch padding is not 0, MindSpore needs to set pad_mode to "pad" mode. If PyTorch sets padding to (2, 3), MindSpore needs to set padding to (2, 2, 3, 3).

```python
# PyTorch
import torch
import torch.nn.functional as F
import numpy as np

y = torch.tensor(np.ones([10, 32, 32, 32]), dtype=torch.float32)
weight = torch.tensor(np.ones([32, 32, 3, 3]), dtype=torch.float32)
output2 = F.conv2d(y, weight, padding=(2, 3))
print(output2.shape)
# torch.Size([10, 32, 34, 36])

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
output = ops.conv2d(x, weight, pad_mode="pad", padding=(2, 2, 3, 3))
print(output.shape)
# (10, 32, 34, 36)

```
