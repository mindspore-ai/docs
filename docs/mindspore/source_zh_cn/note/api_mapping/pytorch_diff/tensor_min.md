# 比较与torch.Tensor.min的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_min.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|   PyTorch APIs   |    MindSpore APIs    |
|:----------------:|:--------------------:|
|    torch.min     |  mindspore.ops.min   |
| torch.Tensor.min | mindspore.Tensor.min |

## torch.Tensor.min

```python
torch.Tensor.min(dim=None,
                 keepdim=False
                 )
```

更多内容详见[torch.Tensor.min](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.min)。

## mindspore.Tensor.min

```python
mindspore.Tensor.min(axis=None,
                     keepdims=False,
                     initial=None,
                     where=True)
```

更多内容详见[mindspore.Tensor.min](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.min.html)。

## 差异对比

MindSpore在PyTorch的基础上，兼容了Numpy的入参 `initial` 和 `where` 。

| 分类  | 子类  | PyTorch | MindSpore | 差异         |
|-----|-----|---------|-----------|------------|
| 输入  | 输入1 | dim     | axis      | 功能一致，参数名不同 |
|     | 输入2 | keepdim | keepdims  | 功能一致，参数名不同 |
|     | 输入3 | -      |initial    | 不涉及        |
|     | 输入4 |  -     |where    | 不涉及        |

### 代码示例1

两API实现功能一致，MindSpore包含了Numpy的扩展。

```python
# PyTorch
import torch
from torch import tensor

a = tensor([[0.6750, 1.0857, 1.7197]])
output = a.min()
# tensor(0.6750)

# MindSpore
import mindspore
from mindspore import Tensor

a = Tensor([[0.6750, 1.0857, 1.7197]])
output = a.min()
print(output)
# 0.675
```
