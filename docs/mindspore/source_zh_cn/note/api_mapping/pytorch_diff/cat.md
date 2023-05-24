# 比较与torch.cat的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cat.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cat

```text
torch.cat(
    tensors,
    dim=0,
    *,
    out=None
) -> Tensor
```

更多内容详见[torch.cat](https://pytorch.org/docs/1.8.1/generated/torch.cat.html)。

## mindspore.ops.cat

```text
mindspore.ops.cat(tensors, axis=0) -> Tensor
```

更多内容详见[mindspore.ops.cat](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cat.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

PyTorch：在指定轴上拼接输入Tensors。输入Tensors的精度不一致时，低精度Tensor会自动转成高精度Tensor。

MindSpore：当前要求输入Tensors的数据类型及精度保持一致。在输入Tensor的精度不一致时，可通过ops.cast把低精度Tensor转成高精度Tensor再调用concat算子。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | tensors  | tensors | MindSpore中tensors序列中的各Tensor精度必须保持一致，PyTorch中tensors序列中的各Tensor的精度可以不同 |
|参数 | 参数1 | dim | axis | 参数名不一致 |
|  | 参数2  | out | - | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

### 代码示例

> MindSpore当前要求输入Tensors的数据类型及精度保持一致，若不一致时可通过ops.cast把低精度Tensor转成高精度Tensor再调用concat算子。

```python
# PyTorch
import torch

torch_x1 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float32)
torch_x2 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float32)
torch_x3 = torch.Tensor([[0, 1], [2, 3]]).type(torch.float16)

torch_output = torch.cat((torch_x1, torch_x2, torch_x3))
print(torch_output.numpy())
# [[0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

# In MindSpore，converting low precision to high precision is needed before cat.
ms_x1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
ms_x2 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
ms_x3 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float16))

ms_x3 = mindspore.ops.cast(ms_x2, mindspore.float32)
output = mindspore.ops.cat((ms_x1, ms_x2, ms_x3))
print(output)
# [[0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]
#  [0. 1.]
#  [2. 3.]]
```
