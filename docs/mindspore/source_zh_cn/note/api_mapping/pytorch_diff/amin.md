# 比较与torch.amin的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/amin.md)

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.amin     |  mindspore.ops.amin   |
|   torch.Tensor.amin    |   mindspore.Tensor.amin    |

## torch.amin

```text
torch.amin(input, dim, keepdim=False, *, out=None) -> Tensor
```

更多内容详见[torch.amin](https://pytorch.org/docs/1.8.1/generated/torch.amin.html#torch.amin)。

## mindspore.ops.amin

```text
mindspore.ops.amin(x, axis=(), keepdims=False) -> Tensor
```

更多内容详见[mindspore.ops.amin](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.amin.html)。

## 差异对比

PyTorch：根据指定 `dim`，求 `input` 的最小值元素。`keepdim` 控制输出和输入的维度是否相同。`out` 可以获取输出。

MindSpore：根据指定 `axis`，求 `x` 的最小值元素。`keepdims` 功能和PyTorch一致。MindSpore没有 `out` 参数。MindSpore的 `axis` 有默认值，在 `axis` 是默认值情况下，求 `x` 所有元素的最小值。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | dim   | axis      | MindSpore的 `axis` 有默认值，PyTorch的 `dim` 没有默认值 |
|      | 参数3 | keepdim   | keepdims | 功能一致，参数名不同 |
|      | 参数4 | out   | -         | PyTorch的 `out` 可以获取输出，MindSpore无此参数 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.float32)
print(torch.amin(input, dim=0, keepdim=True))
# tensor([[1., 2., 1.]])

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2, 3], [3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.amin(x, axis=0, keepdims=True))
# [[1. 2. 1.]]
```
