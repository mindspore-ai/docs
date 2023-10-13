# Differences with torch.randint_like

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/randint_like.md)

## torch.randint_like

```text
torch.randint_like(input, low=0, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

For more information, see [torch.randint_like](https://pytorch.org/docs/1.8.1/generated/torch.randint_like.html#randint_like).

## mindspore.ops.randint_like

```text
mindspore.ops.randint_like(input, low, high, *, dtype=None, seed=None)
```

For more information, see [mindspore.ops.randint_like](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.randint_like.html#mindspore.ops.randint_like).

## Differences

PyTorch: `low` is an optional input, and the default value is 0.

MindSpore: `low` is a mandatory input with no default value.

| Categories | Subcategories |PyTorch | MindSpore | Difference                                    |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| Parameters  | Parameter 1 | input         | input     | No difference                                  |
|   | Parameter 2 | low           | low       | `low` in PyTorch has a default value of 0, while `low` in MindSpore does not have a default value |
|   | Parameter 3 | high          | high      | No difference                                  |
|   | Parameter 4 | dtype         | dtype     | No difference                                  |
|   | Parameter 5 | layout        | -         | Common differences                                 |
|   | Parameter 6 | device        | -         | Common differences                                 |
|   | Parameter 7 | requires_grad | -         | Common differences                                 |
|   | Parameter 8 | memory_format | -         | Common differences                                 |
|   | Parameter 9 | -             | seed      | Common differences                                 |

### Code Example

```python
# PyTorch
import torch

# PyTorch does not need to input the value of low, which is equivalent to low=0 in MindSpore.
x = torch.tensor([[2, 3], [1, 2]], dtype=torch.int32)
y = torch.randint_like(x, 10)
print(tuple(y.shape))
# (2, 2)

# MindSpore
import mindspore

# MindSpore must take the default value of low in the torch (0 in this case) and pass it in as input.
x = mindspore.Tensor([[2, 3], [1, 2]], mindspore.int32)
x = mindspore.ops.randint_like(x, 0, 10)
print(x.shape)
# (2, 2)
```
