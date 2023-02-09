# Function Differences with torch.svd

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/svd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> For the functional differences between `mindspore.Tensor.svd` and `torch.Tensor.svd` , refer to the functional differences between `mindspore.ops.svd` and `torch.svd` .

## torch.svd

```python
torch.svd(input, some=True, compute_uv=True, *, out=None)
```

For more information, see [torch.svd](https://pytorch.org/docs/1.8.1/generated/torch.svd.html).

## mindspore.ops.svd

```python
mindspore.ops.svd(a, full_matrices=False, compute_uv=True)
```

For more information, see [mindspore.ops.svd](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.svd.html).

## Differences

PyTorch:

- If `some` is True, the method returns the reduced singular value decomposition.

- If `compute_uv` is True, the order of output values is u, s, v.

MindSpore:

- If `full_matrices` is False, the method returns the reduced singular value decomposition.

- If `compute_uv` is True, the order of output values is s, u, v.

> `torch.svd()` has been deprecated in PyTorch 1.8.0 and later, and alternative api `torch.linalg.svd()` is recommended, which has the same parameter `full_matrices` as `mindspore.ops.svd`.

There is no difference in function.

## Code Example

> The output values of singular value decomposition are not unique.

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
