# 比较与torch.take的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/take.md)

## torch.Tensor.take

```python
torch.Tensor.take(indices)
```

更多内容详见[torch.Tensor.take](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.take)。

## mindspore.Tensor.take

```python
mindspore.Tensor.take(indices, axis=None, mode='clip')
```

更多内容详见[mindspore.Tensor.take](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.take.html)。

## 使用方式

MindSpore此API功能与PyTorch基本一致。

PyTorch：获取Tensor中的元素。不可指定维度，使用展开的输入数组。若索引超出范围，则抛出异常。

MindSpore：在指定维度上获取Tensor中的元素。可以指定维度，默认使用展开的输入数组。若索引超出范围：mode为'raise'时，则抛出异常；mode为'wrap'时，绕接；mode为'raise'时，裁剪到范围。

| 分类       | 子类         | PyTorch      | MindSpore  | 差异          |
| ---------- | ------------ | ------------ | ---------  | ------------- |
| 参数       | 参数 1       | indices        | indices   |  无  |
|            | 参数 2       |               | axis       | 指定获取的索引，Pytorch不支持 |
|            | 参数 3       |               | mode       | 若索引超出范围时的行为模式选择，Pytorch不支持 |

## 代码示例 1

```python
# PyTorch
import torch
input_x1 = torch.tensor([[4, 3, 5], [6, 7, 8]])
indices = torch.tensor([0, 2, 4])
output = input_x1.take(indices)
print(output)
# tensor([4, 5, 7])

# MindSpore
import mindspore as ms
input_x1 = ms.Tensor([[4, 3, 5], [6, 7, 8]])
indices = ms.Tensor([0, 2, 4])
output = input_x1.take(indices)
print(output)
# [4 5 7]
```

## 代码示例 2

```python
# PyTorch
import torch
input_x1 = torch.tensor([[4, 3, 5], [6, 7, 8]])
indices = torch.tensor([0, 2, 8])
output = input_x1.take(indices)
print(output)
# IndexError: out of range: tried to access index 8 on a tensor of 6 elements

# MindSpore
import mindspore as ms
input_x1 = ms.Tensor([[4, 3, 5], [6, 7, 8]])
indices = ms.Tensor([0, 2, 8])
output = input_x1.take(indices, mode='clip')
print(output)
# [4 5 8]
```
