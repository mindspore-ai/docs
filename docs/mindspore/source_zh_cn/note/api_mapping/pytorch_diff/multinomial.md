# 比较与torch.multinomial的差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/multinomial.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.ops.multinomial](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.multinomial.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

MindSpore: 参数 `replacement` 的默认值为 ``True`` ，即每次采样后把采样的数据放回。

PyTorch: 参数 `replacement` 的默认值为 ``False`` ，即每次采样后不把采样的数据放回。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input         | 一致           |
|            | 参数 2       | num_samples   | num_samples   | 一致           |
|            | 参数 3       | replacement   | replacement   | PyTorch的默认值为 ``False`` ，MindSpore的默认值为 ``True`` |
|            | 参数 4       | generator     | seed          | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r1.11/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|            | 参数 5       | out           | -             | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r1.11/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例

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
