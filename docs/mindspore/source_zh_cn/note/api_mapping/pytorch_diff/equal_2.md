# 比较与torch.equal的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/equal_2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.equal    |   mindspore.ops.equal    |
|    torch.Tensor.equal     |  mindspore.Tensor.equal   |

## torch.equal

```text
torch.equal(input, other) -> bool
```

更多内容详见[torch.equal](https://pytorch.org/docs/1.8.1/generated/torch.equal.html)。

## mindspore.ops.equal

```text
mindspore.ops.equal(x, y) -> Tensor
```

更多内容详见[mindspore.ops.equal](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.equal.html)。

## 差异对比

PyTorch：如果两个输入Tensor的size和元素均相等，返回True, 否则返回False。

MindSpore：逐元素比较两个输入Tensor是否相等。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| :-: | :-: | :-: | :-: |:-:|
|参数 | 参数1 | input | x | 参数名不同 |
| | 参数2 | other | y | 参数名不同 |

### 代码示例

> torch.equal与mindspore.ops.equal的实现功能不一致。torch.equal比较两个输入Tensor的size和其中的元素是否相等，返回值为bool类型；mindspore.ops.equal逐元素比较两个输入Tensor是否相等，返回值为Tensor，其shape与输入Tensor广播后的shape相同，数据类型为bool。

```python
# PyTorch
import torch
from torch import tensor

input1 = tensor([1, 2], dtype=torch.float32)
other = tensor([[1, 2], [0, 2], [1, 3]], dtype=torch.int64)
out = torch.equal(input1, other).numpy()
print(out)
# False

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]), mindspore.float32)
y = Tensor(np.array([[1, 2], [0, 2], [1, 3]]), mindspore.int64)
output = mindspore.ops.equal(x, y)
print(output)
# [[ True  True]
#  [False  True]
#  [ True False]]
```