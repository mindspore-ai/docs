# Function Differences with torch.hann_window

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.numpy.hanning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.hann_window

```text
torch.hann_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

For more information, see [torch.hann_window](https://pytorch.org/docs/1.8.1/generated/torch.hann_window.html).

## mindspore.numpy.hanning

```text
mindspore.numpy.hanning(M) -> Tensor
```

For more information, see [mindspore.numpy.hanning](https://mindspore.cn/docs/en/master/api_python/numpy/mindspore.numpy.hanning.html).

## Differences

PyTorch: Return the Hanning window with the same size as window_length. The periodic parameter determines whether the returned window will remove the last duplicate value of the symmetric window.

MindSpore: MindSpore API basically implements the same function as PyTorch, but it lacks the parameter periodic, which is equivalent to setting periodic to False.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Input | Single input |window_length | M | Same function, different parameter names |
|Parameters | Parameter 1 | periodic | -    | Equivalent to setting periodic to False in MindSpore |
|  | Parameter 2 | dtype        | -    | MindSpore does not have this parameter, and the output dtype is Float32, consistent with the default of the marker |
| | Parameter 3 | layout | - | Not involved |
| | Parameter 4 | device | - | Not involved |
| | Parameter 5 | requires_grad | - | MindSpore does not have this parameter and supports reverse derivation by default |

### Code Example 1

> The periodic parameter in the PyTorch operator determines whether the return window will remove the last repeated value of the symmetric window, while the MindSpore operator lacks the parameter, equivalent to setting periodic to False.

```python
# PyTorch
import torch

torch_output = torch.hann_window(12, periodic=False)
print(torch_output.numpy())
# [0.         0.07937324 0.29229248 0.57115734 0.82743037 0.97974646
#  0.9797465  0.8274305  0.5711575  0.29229265 0.07937327 0.        ]

# MindSpore
import mindspore

ms_output = mindspore.numpy.hanning(12)
print(ms_output)
# [0.         0.07937324 0.29229248 0.57115734 0.8274303  0.97974694
#  0.97974706 0.8274305  0.5711576  0.29229274 0.07937327 0.        ]
```
