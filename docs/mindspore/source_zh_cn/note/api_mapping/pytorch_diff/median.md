# 比较与torch.median的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/median.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `mindspore.Tensor.median` 和 `torch.Tensor.median` 的功能差异，参考 `mindspore.ops.median` 和 `torch.median` 的功能差异比较。

## torch.median

```text
torch.median(input, dim=-1, keepdim=False, *, out=None) -> Tensor
```

更多内容详见[torch.median](https://pytorch.org/docs/1.8.1/generated/torch.median.html#torch.median)。

## mindspore.ops.median

```text
mindspore.ops.median(x, axis=-1, keepdims=False) -> Tensor
```

更多内容详见[mindspore.ops.median](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.median.html)。

## 差异对比

PyTorch：根据指定 `dim`，输出 `input` 的中值与索引。`keepdim` 控制输出和输入的维度是否相同。在输入仅有 `input` 时，返回所有元素的中值；在输入包含 `dim` 时，返回指定维度的中值和索引。`out` 可以获取输出。

MindSpore：根据指定 `axis`，输出 `x` 的中值与索引。`keepdims` 功能和PyTorch一致。与Pytorch不同，不论输入包含不包含 `axis`，MindSpore返回指定维度上的中值与索引。MindSpore没有 `out` 参数。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | dim   | axis      | 功能一致，参数名不同 |
|      | 参数3 | keepdim   | keepdims | 功能一致，参数名不同 |
|      | 参数4 | out   | -         | PyTorch的 `out` 可以获取输出，MindSpore无此参数 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=torch.float32)
print(torch.median(input))
# tensor(2.)
print(torch.median(input, dim=1, keepdim=True))
# torch.return_types.median(
# values=tensor([[1.],
#         [2.]]),
# indices=tensor([[3],
#         [2]]))

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.median(x, axis=1, keepdims=True))
# (Tensor(shape=[2, 1], dtype=Float32, value=
# [[ 1.00000000e+00],
#  [ 2.00000000e+00]]), Tensor(shape=[2, 1], dtype=Int64, value=
# [[3],
#  [2]]))
```
