# 比较与torch.prod的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/prod.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `mindspore.Tensor.prod` 和 `torch.Tensor.prod` 的功能差异，参考 `mindspore.ops.prod` 和 `torch.prod` 的功能差异比较。

## torch.prod

```text
torch.prod(input, dim, keepdim=False, *, dtype=None) -> Tensor
```

更多内容详见[torch.prod](https://pytorch.org/docs/1.8.1/generated/torch.prod.html#torch.prod)。

## mindspore.ops.prod

```text
mindspore.ops.prod((x, axis=(), keep_dims=False) -> Tensor
```

更多内容详见[mindspore.ops.prod](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.prod.html)。

## 差异对比

PyTorch：根据指定 `dim`，对 `input` 中元素求乘积。`keepdim` 控制输出和输入的维度是否相同。`dtype` 设置输出Tensor的数据类型。

MindSpore：根据指定 `axis`，对 `x` 中元素求乘积。`keep_dims` 功能和PyTorch一致。MindSpore没有 `dtype` 参数。MindSpore的 `axis` 有默认值，在 `axis` 是默认值情况下，对 `x` 所有元素求乘积。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | dim   | axis      | PyTorch必须传入 `dim` 且只能传入一个整数，MindSpore的 `axis` 可以传入整数，整数的tuple或整数的list |
|      | 参数3 | keepdim   | keep_dims | 功能一致，参数名不同 |
|      | 参数4 | dtype   | -         | PyTorch的 `dtype` 可以设置输出Tensor的数据类型，MindSpore无此参数 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=torch.float32)
print(torch.prod(input, dim=1, keepdim=True))
# tensor([[ 7.5000],
#         [15.0000]])
print(torch.prod(input, dim=1, keepdim=True, dtype=torch.int32))
# tensor([[ 6],
#         [12]], dtype=torch.int32)

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2.5, 3, 1], [2.5, 3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.prod(x, axis=1, keep_dims=True))
# [[ 7.5]
#  [15. ]]
```
