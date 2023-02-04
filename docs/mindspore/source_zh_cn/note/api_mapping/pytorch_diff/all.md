# 比较与torch.all的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.all

```text
torch.all(input, dim, keepdim=False, *, out=None) -> Tensor
```

更多内容详见[torch.all](https://pytorch.org/docs/1.8.1/generated/torch.all.html#torch.all)。

## mindspore.ops.all

```text
mindspore.ops.all((x, axis=(), keep_dims=False) -> Tensor
```

更多内容详见[mindspore.ops.all](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.all.html)。

## 差异对比

PyTorch：根据指定 `dim`，对 `input` 的元素进行逻辑与。`keepdim` 控制输出和输入的维度是否相同。`out` 可以获取输出。

MindSpore：根据指定 `axis`，对 `x` 的元素进行逻辑与。`keep_dims` 功能和PyTorch一致。MindSpore没有 `out` 参数。MindSpore的 `axis` 有默认值，在 `axis` 是默认值情况下，对 `x` 所有元素进行逻辑与。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | dim   | axis      | PyTorch必须传入 `dim` 且只能传入一个整数，MindSpore的 `axis` 可以传入整数，整数的tuple或整数的list |
|      | 参数3 | keepdim   | keep_dims | 功能一致，参数名不同 |
|      | 参数4 | out   | -         | PyTorch的 `out` 可以获取输出，MindSpore无此参数 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[False, True, False, True], [False, True, False, False]])
print(torch.all(input, dim=0, keepdim=True))
# tensor([[False,  True, False, False]])

# MindSpore
import mindspore

x = mindspore.Tensor([[False, True, False, True], [False, True, False, False]])
print(mindspore.ops.all(x, axis=0, keep_dims=True))
# [[False  True False False]]
```
