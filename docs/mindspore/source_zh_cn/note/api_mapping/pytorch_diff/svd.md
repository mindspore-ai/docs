# 比较与torch.svd的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/svd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

以下映射关系均可参考本文。

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.svd    |   mindspore.ops.svd    |
|    torch.Tensor.svd   |  mindspore.Tensor.svd   |

## torch.svd

```python
torch.svd(input, some=True, compute_uv=True, *, out=None)
```

更多内容详见[torch.svd](https://pytorch.org/docs/1.8.1/generated/torch.svd.html)。

## mindspore.ops.svd

```python
mindspore.ops.svd(input, full_matrices=False, compute_uv=True)
```

更多内容详见[mindspore.ops.svd](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.svd.html)。

## 差异对比

MindSpore此API功能与PyTorch不一致。

PyTorch:

- 如果参数 `some` 为True，该接口返回缩减后的奇异值分解结果。

- 始终有三个输出值，输出值的顺序是 u，s，v。

- 如果参数 `compute_uv` 为False，u 和 v 的值是全0的矩阵。

MindSpore:

- 如果参数 `full_matrices` 为False，该接口返回缩减后的奇异值分解结果。

- 如果参数 `compute_uv` 为False，只有一个输出值 s。

- 如果参数 `compute_uv` 为True，有三个输出值，顺序是 s，u，v。

> 自PyTorch 1.8.0及以后的版本中，已经弃用了接口 `torch.svd()` ，推荐使用的替换接口是 `torch.linalg.svd()` ，该接口和 `mindspore.ops.svd` 有相同的传参 `full_matrices` 。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | input         | input         | 一致           |
|            | 参数 2       | some          | full_matrices | 若要返回缩减后的奇异值分解结果，MindSpore配置 `full_matrices` 为False，PyTorch配置 `some` 为True |
|            | 参数 3       | compute_uv    | compute_uv    | 如果参数 `compute_uv` 为False，MindSpore只有一个输出值 s，PyTorch有三个输出值 u，s，v，其中 u 和 v 的值是全0的矩阵。如果 `compute_uv` 为True，MindSpore的输出值的顺序是 s，u，v，PyTorch的输出值的顺序是 u，s，v。 |
|            | 参数 4       | out           | -             | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例 1

> `compute_uv` 为False时，PyTorch有三个输出值。

```python
# PyTorch
import torch
input = torch.tensor([[1, 2], [-4, -5], [2, 1]], dtype=torch.float32)
u, s, v = torch.svd(input, some=False, compute_uv=False)
print(s)
# tensor([7.0653, 1.0401])
print(u)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])
print(v)
# tensor([[0., 0.],
#         [0., 0.]])

# MindSpore目前无法支持该功能
```

## 代码示例 2

> `compute_uv` 为True的时候，输出值顺序不一致。
> 奇异值分解的输出不是唯一的。

```python
# PyTorch
import torch
input = torch.tensor([[1, 2], [-4, -5], [2, 1]], dtype=torch.float32)
u, s, v = torch.svd(input, some=False, compute_uv=True)
print(s)
# tensor([7.0653, 1.0401])
print(u)
# tensor([[-0.3082, -0.4882,  0.8165],
#         [ 0.9061,  0.1107,  0.4082],
#         [-0.2897,  0.8657,  0.4082]])
print(v)
# tensor([[-0.6386,  0.7695],
#         [-0.7695, -0.6386]])

# MindSpore
import mindspore as ms
input = ms.Tensor([[1, 2], [-4, -5], [2, 1]], ms.float32)
s, u, v = ms.ops.svd(input, full_matrices=True, compute_uv=True)
print(s)
# [7.0652843 1.040081 ]
print(u)
# [[ 0.30821905 -0.48819482  0.81649697]
#  [-0.90613353  0.11070572  0.40824813]
#  [ 0.2896955   0.8656849   0.4082479 ]]
print(v)
# [[ 0.63863593  0.769509  ]
#  [ 0.769509   -0.63863593]]
```
