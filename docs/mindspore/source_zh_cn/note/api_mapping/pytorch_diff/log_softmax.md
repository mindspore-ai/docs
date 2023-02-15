# 比较与torch.nn.functional.log_softmax的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/log_softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.log_softmax

```python
torch.nn.functional.log_softmax(
    input,
    dim=None,
    _stacklevel=3,
    dtype=None
)
```

更多内容详见[torch.nn.functional.log_softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.log_softmax)。

## mindspore.ops.log_softmax

```python
class mindspore.ops.log_softmax(
    logits,
    axis=-1,
)
```

更多内容详见[mindspore.ops.log_softmax](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.log_softmax.html)。

## 差异对比

PyTorch：支持使用`dim`参数和`input`输入实现函数，对softmax的结果取对数。

MindSpore：支持使用`axis`参数和`logits`输入实现函数，对softmax的结果取对数。

| 分类 | 子类  | PyTorch | MindSpore | 差异                    |
| ---- | ----- | ------ | --------- | ----------------------- |
| 参数 | 参数1 | input  | logits    | 功能一致，参数名不同 |
|      | 参数2 | dim  | axis | 功能一致，参数名不同 |
|      | 参数3 | dtype | - | PyTorch中用来指定输出Tensor的data type，MindSpore中没有该参数 |

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, we can define an instance of this class first, and the default value of the parameter axis is -1.
x = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
output1 = ops.log_softmax(x)
print(output1)
# Out:
# [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
x = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
output2 = ops.log_softmax(x, axis=0)
print(output2)
# out:
# [[0.01798621 0.11920292 0.5        0.880797   0.98201376]
#  [0.98201376 0.880797   0.5        0.11920292 0.01798621]]

# In torch, the input and dim should be input at the same time to implement the function.
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.functional.log_softmax(input, dim=0)
print(output3)
# Out:
# tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364], dtype=torch.float64)
```
