# Function Differences with torch.randint_like

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/randint_like.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.randint_like

```text
torch.randint_like(input, low=0, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

For more information,
see [torch.randint_like](https://pytorch.org/docs/1.8.1/generated/torch.randint_like.html#torch.randint_like).

## mindspore.ops.randint_like

```text
mindspore.ops.randint_like(input, low, high, *, dtype=None, seed=None)
```

For more information,
see [mindspore.ops.randint_like](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randint_like.html#mindspore.ops.randint_like).

## Differences

PyTorch：`low` is optional with default value: 0.

MindSpore：`low` is not optional and has no default value.

| Categories | Subcategories | PyTorch       | MindSpore | Differences                                               |
|------------|---------------|---------------|-----------|-----------------------------------------------------------|
| Parameter  | Parameter 1   | input         | input     | -                                                         |
|            | Parameter 2   | low           | low       | `low` has default value 0 in PyTorch, MindSpore does not. |
|            | Parameter 3   | high          | high      | -                                                         |
|            | Parameter 4   | dtype         | dtype     | -                                                         |
|            | Parameter 5   | layout        | -         | General difference                                        |
|            | Parameter 6   | device        | -         | General difference                                        |
|            | Parameter 7   | requires_grad | -         | General difference                                        |
|            | Parameter 8   | memory_format | -         | General difference                                        |
|            | Parameter 9   | -             | seed      | General difference                                        |

### Code Example 1

```python
# PyTorch
import torch

# PyTorch does not need to set the value of low.
x = torch.tensor([[2, 3], [1, 2]], dtype=torch.float32)
y = torch.randint_like(x, 10)
print(tuple(y.shape))
# (2, 2)

# MindSpore
import mindspore

# MindSpore must set the default value of low in PyTorch(0 in this case), as one of the inputs.
x = mindspore.Tensor([[2, 3], [1, 2]], mindspore.float32)
y = mindspore.ops.randint_like(x, 0, 10)
print(y.shape)
# (2, 2)
```