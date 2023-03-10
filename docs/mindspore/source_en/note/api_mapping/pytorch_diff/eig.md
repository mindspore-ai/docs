# Function Differences with torch.eig

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/eig.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.eig

```text
torch.eig(input, eigenvectors=False, *, out=None) -> Tensor
```

For more information, see [torch.eig](https://pytorch.org/docs/1.8.1/generated/torch.eig.html#torch.eig).

## mindspore.ops.eig

```text
mindspore.ops.eig(A) -> Tensor
```

For more information, see [mindspore.ops.eig](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.eig.html).

## Differences

PyTorch: if `eigenvectors`is True, return the `eigenvalues` and `eigenvectors`, otherwise only return `eigenvalues`. In 1.9 and later version, `torch.eig` has been replaced by `torch.linalg.eig`, `mindspore.ops.eig` is consistent with `torch.linalg.eig`.

MindSpore: returns `eigenvalues` and `eigenvectors`。

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | input   | A         | Same function, different parameter names                    |
|      | Parameter 2 | eigenvectors   | -      | Not involved |
|      | Parameter 3 | out   | -         | PyTorch `out` can get the output. MindSpore does not have this parameter |

### Code Exampless

```python
# PyTorch
import torch

inputs = np.array([[1.0, 0.0], [0.0, 2.0]])
value, vector = torch.eig(torch.tensor(inputs))
print(value)
# [1.+0.j 2.+0.j]
print(vector)
# [[1.+0.j 0.+0.j]
#  [0.+0.j 1.+0.j]]

# MindSpore
import mindspore

value, vector = mindspore.ops.eig(Tensor(inputs, mindspore.float32))
print(value)
# [1.+0.j 2.+0.j]
print(vector)
# [[1.+0.j 0.+0.j]
```
