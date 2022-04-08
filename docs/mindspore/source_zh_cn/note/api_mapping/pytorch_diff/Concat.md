# 比较与torch.cat的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Concat.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cat

```python
torch.cat(
    tensors,
    dim=0,
    out=None
)
```

更多内容详见[torch.cat](https://pytorch.org/docs/1.5.0/torch.html#torch.cat)。

## mindspore.ops.Concat

```python
class mindspore.ops.Concat(
    axis=0
)(input_x)
```

更多内容详见[mindspore.ops.Concat](https://mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Concat.html#mindspore.ops.Concat)。

## 使用方式

PyTorch: 输入tensor的数据类型不同时，低精度tensor会自动转成高精度tensor。

MindSpore: 当前要求输入tensor的数据类型保持一致，若不一致时可通过ops.Cast把低精度tensor转成高精度类型再调用Concat算子。

## 代码示例

```python
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import torch
import numpy as np

# In MindSpore，converting low precision to high precision is needed before concat.
a = Tensor(np.ones([2, 3]).astype(np.float16))
b = Tensor(np.ones([2, 3]).astype(np.float32))
concat_op = ops.Concat()
cast_op = ops.Cast()
output = concat_op((cast_op(a, mindspore.float32), b))
print(output.shape)
# Out：
# (4, 3)

# In Pytorch.
a = torch.tensor(np.ones([2, 3]).astype(np.float16))
b = torch.tensor(np.ones([2, 3]).astype(np.float32))
output = torch.cat((a, b))
print(output.size())
# Out：
# torch.Size([4, 3])
```