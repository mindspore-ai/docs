# 比较与torch.multinomial的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/multinomial.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs          |      MindSpore APIs           |
| :-------------------:     | :-----------------------:     |
| torch.multinomial         | mindspore.ops.multinomial     |
| torch.Tensor.multinomial  | mindspore.Tensor.multinomial  |

## torch.multinomial

```python
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
```

更多内容详见[torch.multinomial](https://pytorch.org/docs/1.8.1/generated/torch.multinomial.html)。

## mindspore.ops.multinomial

```python
mindspore.ops.multinomial(input, num_samples, replacement=True, seed=None)
```

更多内容详见[mindspore.ops.multinomial](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.multinomial.html)。

## 差异对比

MindSpore下的参数名和默认值和PyTorch存在差异，功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input         | 一致           |
|            | 参数 2       | num_samples   | num_samples   | 一致           |
|            | 参数 3       | replacement   | replacement   | 功能一致，默认值不同。PyTorch的默认值为False，MindSpore的默认值为True |
|            | 参数 4       | generator     | seed          | MindSpore使用随机数种子生成随机数 |
|            | 参数 5       | out           | -             | 不涉及        |

## 差异分析与示例

```python
# PyTorch
import torch

input = torch.tensor([0, 9, 4, 0], dtype=torch.float32)
output = torch.multinomial(input, 2)
print(output)
# tensor([1, 2]) or tensor([2, 1])

# MindSpore
import mindspore as ms

input = ms.Tensor([0, 9, 4, 0], dtype=ms.float32)
output = ms.ops.multinomial(input, 2, False)
print(output)
# [1 2] or [2 1]
```
