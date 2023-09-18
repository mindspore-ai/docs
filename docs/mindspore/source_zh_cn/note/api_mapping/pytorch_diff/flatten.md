# 比较与torch.flatten的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/flatten.md)

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
mindspore.ops.flatten(input, order='C', *, start_dim=1, end_dim=-1)
```

更多内容详见[mindspore.ops.flatten](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.flatten.html)。

## 使用方式

PyTorch：支持指定维度对元素进行展开，`start_dim` 默认为0，`end_dim` 默认为-1。

MindSpore：支持指定维度对元素进行展开，`start_dim` 默认为1，`end_dim` 默认为-1。通过 `order` 为"C"或"F"确定优先按行还是列展平。

| 分类  | 子类  | PyTorch   | MindSpore | 差异         |
|-----|-----|-----------|-----------|------------|
| 参数 | 参数1 | input     | input      | 功能一致        |
|     | 参数2 | -         | order      | 展平优先顺序选项，PyTorch无此参数        |
|     | 参数3 | start_dim | start_dim  | 功能一致        |
|     | 参数4 | end_dim   | end_dim    | 功能一致        |

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# MindSpore
input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
output = ops.flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
output = ops.flatten(input_tensor, start_dim=2)
print(output.shape)
# Out：
# (1, 2, 12)

# PyTorch
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
