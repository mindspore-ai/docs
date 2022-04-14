# 比较与torch.nn.GELU的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/FastGelu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.GELU

```python
class torch.nn.GELU()(input)
```

更多内容详见[torch.nn.GELU](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.GELU)。

## mindspore.nn.FastGelu

```python
class mindspore.nn.FastGelu()(input_data)
```

更多内容详见[mindspore.nn.FastGelu](https://mindspore.cn/docs/zh-CN/r1.7/api_python/nn/mindspore.nn.FastGelu.html#mindspore.nn.FastGelu)。

## 使用方式

PyTorch：基于高斯分布的累积分布函数。

MindSpore：采用与PyTorch不同的计算公式。

## 代码示例

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