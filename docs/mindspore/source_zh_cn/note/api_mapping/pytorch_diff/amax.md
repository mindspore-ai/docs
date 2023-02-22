# 比较与torch.amax的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/amax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.amax     |  mindspore.ops.amax   |
|   torch.Tensor.amax    |   mindspore.Tensor.amax    |

## torch.amax

```text
torch.amax(input, dim, keepdim=False, *, out=None) -> Tensor
```

更多内容详见[torch.amax](https://pytorch.org/docs/1.8.1/generated/torch.amax.html#torch.amax)。

## mindspore.ops.amax

```text
mindspore.ops.amax(x, axis=(), keep_dims=False) -> Tensor
```

更多内容详见[mindspore.ops.amax](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.amax.html)。

## 差异对比

PyTorch：根据指定 `dim`，求 `input` 的最大值元素。`keepdim` 控制输出和输入的维度是否相同。`out` 可以获取输出。

MindSpore：根据指定 `axis`，求 `x` 的最大值元素。`keep_dims` 功能和PyTorch一致。MindSpore没有 `out` 参数。MindSpore的 `axis` 有默认值，在 `axis` 是默认值情况下，求 `x` 所有元素的最大值。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同                    |
|      | 参数2 | dim   | axis      | MindSpore的 `axis` 有默认值，PyTorch的 `dim` 没有默认值 |
|      | 参数3 | keepdim   | keep_dims | 功能一致，参数名不同 |
|      | 参数4 | out   | -         | PyTorch的 `out` 可以获取输出，MindSpore无此参数 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.float32)
print(torch.amax(input, dim=0, keepdim=True))
# tensor([[3., 2., 3.]])

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2, 3], [3, 2, 1]], dtype=mindspore.float32)
print(mindspore.ops.amax(x, axis=0, keep_dims=True))
# [[3. 2. 3.]]
```
