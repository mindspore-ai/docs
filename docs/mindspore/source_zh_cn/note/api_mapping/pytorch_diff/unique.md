# 比较与torch.unique的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/unique.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.unique

```python
torch.unique(
    *args,
    **kwargs
)
```

更多内容详见[torch.unique](https://pytorch.org/docs/1.8.1/generated/torch.unique.html#torch.unique)。

## mindspore.ops.unique

```python
mindspore.ops.unique(x)
```

更多内容详见[mindspore.ops.unique](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.unique.html#mindspore.ops.unique)。

## 差异对比

PyTorch：对Tensor中元素进行去重。可通过设置参数 `sorted` 确定输出是否按升序排列；设置参数 `return_inverse` 确定是否输出输入的Tensor的各元素在输出Tensor中的位置索引；设置参数 `return_counts` 确定是否输出各唯一值在输入的Tensor中的数量；设置参数 `dim` 指定unique的维度。MindSpore不支持这些功能。

MindSpore：对Tensor中元素进行去重，以及返回输入Tensor的各元素在输出Tensor中的位置索引。

| 分类 | 子类  | PyTorch | MindSpore | 差异                  |
| ---- | ----- | ------- | --------- | --------------------- |
| 参数 | 参数1 | input   | x | 输入Tensor，参数名不同 |
|  | 参数2 | sorted | - | sorted为True时，输出Tensor按照升序排列；sorted为False时，按照原有顺序排列 |
|  | 参数3 | return_inverse | - | return_inverse为True时，返回输入Tensor各元素在输出Tensor中的索引位置 |
|  | 参数4 | return_counts | - | return_counts为True时，返回输出Tensor各元素在输入Tensor中的数量 |
|  | 参数5 | dim | - | 指定unique的维度 |

## 代码示例

```python
# In MindSpore
import mindspore

x = mindspore.Tensor([1, 3, 2, 3], mindspore.float32)
output, idx = mindspore.ops.unique(x)
print(output)
# [1. 3. 2.]
print(idx)
# [0 1 2 1]

# In PyTorch
import torch

output, inverse_indices, counts = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True, return_counts=True)
print(output)
# tensor([1, 2, 3])
print(inverse_indices)
# tensor([0, 2, 1, 2])
print(counts)
# tensor([1, 1, 2])

# Example of using unique with dim
output, inverse_indices = torch.unique(torch.tensor([[3, 1], [1, 2]], dtype=torch.long), sorted=True, return_inverse=True, dim=0)
print(output)
# tensor([[1, 2],
#         [3, 1]])
print(inverse_indices)
# tensor([1, 0])
```