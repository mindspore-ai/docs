# 比较与torch.flatten的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/flatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.flatten

```python
torch.flatten(
    input,
    start_dim=0,
    end_dim=-1
)
```

更多内容详见[torch.flatten](https://pytorch.org/docs/1.8.1/generated/torch.flatten.html)。

## mindspore.ops.flatten

```python
mindspore.ops.flatten(input_x)
```

更多内容详见[mindspore.ops.flatten](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.flatten.html)。

## 使用方式

PyTorch：支持指定维度对元素进行展开。

MindSpore：仅支持保留第零维元素，对其余维度的元素进行展开。

| 分类  | 子类  | PyTorch   | MindSpore | 差异         |
|-----|-----|-----------|-----------|------------|
| 参数  | 参数1 | input     | input_x  | 功能一致，参数名不同 |
|     | 参数2 | start_dim | -         | 不涉及        |
|     | 参数3 | end_dim   | -         | 不涉及        |

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
output = ops.flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

# In torch, the dimension to reserve will be specified and the rest will be flattened.
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output1 = torch.flatten(input=input_tensor, start_dim=1)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output2 = torch.flatten(input=input_tensor, start_dim=2)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```