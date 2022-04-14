# Function Differences with torch.nn.GEL

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/FastGelu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torch.nn.GELU

```python
class torch.nn.GELU()(input)
```

For more information, see [torch.nn.GELU](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.GELU).

## mindspore.nn.FastGelu

```python
class mindspore.nn.FastGelu()(input_data)
```

For more information, see [mindspore.nn.FastGelu](https://mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.FastGelu.html#mindspore.nn.FastGelu).

## Differences

PyTorch: Cumulative distribution function based on Gaussian distribution.

MindSpore：Compared with PyTorch, MindSpore adopts a different calculation formula and has better performance.

## Code Example

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

def test_me():
    input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    fast_gelu = nn.FastGelu()
    output = fast_gelu(input_x)
    print(output)

def test_torch():
    input_x = torch.Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]))
    gelu = torch.nn.GELU()
    output = gelu(input_x)
    print(output)

if __name__ == '__main__':
    test_me()
    test_torch()

# Out：
# [[-1.5419e-01  3.9922e+00 -9.7474e-06]
#  [ 1.9375e+00 -1.0053e-03  8.9824e+00]]
# tensor([[-1.5866e-01,  3.9999e+00, -0.0000e+00],
#         [ 1.9545e+00, -1.4901e-06,  9.0000e+00]])
```