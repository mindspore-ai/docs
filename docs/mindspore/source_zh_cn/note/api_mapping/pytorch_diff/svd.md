# 比较与torch.svd的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/svd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `mindspore.Tensor.svd` 和 `torch.Tensor.svd` 的功能差异，参考 `mindspore.ops.svd` 和 `torch.svd` 的功能差异比较。

## torch.svd

```python
torch.svd(input, some=True, compute_uv=True, *, out=None)
```

更多内容详见[torch.svd](https://pytorch.org/docs/1.8.1/generated/torch.svd.html)。

## mindspore.ops.svd

```python
mindspore.ops.svd(a, full_matrices=False, compute_uv=True)
```

更多内容详见[mindspore.ops.svd](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.svd.html)。

## 差异对比

PyTorch:

- 如果参数 `some` 为True，该接口返回缩减后的奇异值分解结果。

- 如果参数 `compute_uv` 为True输出值的顺序是 u，s，v。

MindSpore:

- 如果参数 `full_matrices` 为False，该接口返回缩减后的奇异值分解结果。

- 如果参数 `compute_uv` 为True输出值的顺序是 s，u，v。

> 自PyTorch 1.8.0及以后的版本中，已经弃用了接口 `torch.svd()` ，推荐使用的替换接口是 `torch.linalg.svd()` ，该接口和 `mindspore.ops.svd` 有相同的传参 `full_matrices` 。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | a             | 功能一致，参数名不同 |
|            | 参数 2       | some          | full_matrices | 若要返回缩减后的奇异值分解结果，MindSpore配置 `full_matrices` 为False，PyTorch配置 `some` 为True |
|            | 参数 3       | compute_uv    | compute_uv    | 如果参数 `compute_uv` 为True，MindSpore的输出值的顺序是 s，u，v，PyTorch的输出值的顺序是 u，s，v。 |
|            | 参数 4       | out           | -             | 不涉及        |

## 差异分析与示例

> 奇异值分解的输出不是唯一的。

```python
# PyTorch
import torch
input_x = torch.tensor([[1, 2], [-4, -5], [2, 1]], dtype=torch.float32)
u, s, v = torch.svd(input_x, some=False, compute_uv=True)
print(s)
print(u)
print(v)
# tensor([7.0653, 1.0401])
# tensor([[-0.3082, -0.4882,  0.8165],
#         [ 0.9061,  0.1107,  0.4082],
#         [-0.2897,  0.8657,  0.4082]])
# tensor([[-0.6386,  0.7695],
#         [-0.7695, -0.6386]])

# MindSpore
import mindspore as ms
input_x = ms.Tensor([[1, 2], [-4, -5], [2, 1]], ms.float32)
s, u, v = ms.ops.svd(input_x, full_matrices=True, compute_uv=True)
print(s)
print(u)
print(v)
# [7.0652843 1.040081 ]
# [[ 0.30821905 -0.48819482  0.81649697]
#  [-0.90613353  0.11070572  0.40824813]
#  [ 0.2896955   0.8656849   0.4082479 ]]
# [[ 0.63863593  0.769509  ]
#  [ 0.769509   -0.63863593]]
```
