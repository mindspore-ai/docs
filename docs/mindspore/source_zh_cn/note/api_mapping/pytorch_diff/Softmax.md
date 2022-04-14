# 比较与torch.nn.functional.softmax的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.softmax

```python
torch.nn.functional.softmax(
    input,
    dim=None,
    _stacklevel=3,
    dtype=None
)
```

更多内容详见[torch.nn.functional.softmax](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.softmax)。

## mindspore.ops.Softmax

```python
class mindspore.ops.Softmax(
    axis=-1,
)(logits)
```

更多内容详见[mindspore.ops.Softmax](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Softmax.html#mindspore.ops.Softmax)。

## 使用方式

PyTorch：支持使用`dim`参数和`input`输入实现函数，将指定维度元素缩放到[0, 1]之间并且总和为1。

MindSpore：支持使用`axis`属性初始化Softmax，将指定维度元素缩放到[0, 1]之间并且总和为1。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, we can define an instance of this class first, and the default value of the parameter axis is -1.
logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
softmax = ops.Softmax()
output1 = softmax(logits)
print(output1)
# Out:
# [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
logits = Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), mindspore.float32)
softmax = ops.Softmax(axis=0)
output2 = softmax(logits)
print(output2)
# out:
# [[0.01798621 0.11920292 0.5        0.880797   0.98201376], [0.98201376 0.880797   0.5        0.11920292 0.01798621]]

# In torch, the input and dim should be input at the same time to implement the function.
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.functional.softmax(input, dim=0)
print(output3)
# Out:
# tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364], dtype=torch.float64)

```
