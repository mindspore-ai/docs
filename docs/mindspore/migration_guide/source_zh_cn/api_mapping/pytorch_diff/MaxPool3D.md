# 比较与torch.nn.MaxPool3d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/MaxPool3D.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.nn.MaxPool3d

```python
torch.nn.MaxPool3d(
    kernel_size=1,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False
)
```

更多内容详见[torch.nn.MaxPool3d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MaxPool3d)。

## mindspore.ops.MaxPool3D

```python
class mindspore.ops.MaxPool3D(
    kernel_size=1,
    strides=1,
    pad_mode='valid',
    pad_list=0,
    ceil_mode=None,
    data_format='NCDHW'
)(input)
```

更多内容详见[mindspore.ops.MaxPool3D](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.MaxPool3D.html#mindspore.ops.MaxPool3D)。

## 使用方式

PyTorch：同时支持五维数据 (N, C, Din, Hin, Win) 和四维数据(C, Din, Hin, Win)。

MindSpore：仅支持五维数据(N, C, Din, Hin, Win)。

迁移建议：如需要MindSpore MaxPool3D处理四维输入，可以用ExpandDims算子将原始输入维度扩张为(1, C, Din, Hin, Win)，传入MaxPool3D后再将输出用Squeeze算子将维度由(1, C, Dout, Hout, Wout)转为(C, Dout, Hout, Wout)。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore
net = ops.MaxPool3D((3, 2, 2), strides=2)
x = Tensor(np.ones([20, 16, 50, 44, 31]), mindspore.float32)
output = net(x).shape
print(output)
# Out:
# (20, 16, 24, 22, 15)

# In PyTorch
m = torch.nn.MaxPool3d((3, 2, 2), stride=2)
input = torch.rand(20, 16, 50, 44, 31)
output = m(input).shape
print(output)
# Out：
# torch.Size([20, 16, 24, 22, 15])
```
