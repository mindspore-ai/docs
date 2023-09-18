# Differences with torch.randint

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/randint.md)

## torch.randint

```text
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

For more information, see [torch.randint](https://pytorch.org/docs/1.8.1/generated/torch.randint.html#torch.randint).

## mindspore.ops.randint

```text
mindspore.ops.randint(low, high, size, seed=None, *, dtype=None)
```

For more information, see [mindspore.ops.randint](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.randint.html#mindspore.ops.randint).

## Differences

PyTorch: `low` is an optional input, and the default value is 0.

MindSpore: `low` is a mandatory input with no default value.

| Categories | Subcategories |PyTorch | MindSpore | Difference                                    |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| Parameters  | Parameter 1  | low           | low       | `low` in PyTorch has a default value of 0, while `low` in MindSpore does not have a default value |
|   | Parameter 2  | high          | high      | No difference                                  |
|   | Parameter 3  | size          | size      | No difference                                  |
|   | Parameter 4  | generator     | -         | Common differences                                 |
|   | Parameter 5  | out           | -         | Common differences                                 |
|   | Parameter 6  | dtype         | dtype     | No difference                                  |
|   | Parameter 7  | layout        | -         | Common differences                                 |
|   | Parameter 8  | device        | -         | Common differences                                 |
|   | Parameter 9  | requires_grad | -         | Common differences                                 |
|   | Parameter 10 | -             | seed      | Common differences                                 |

### Code Example

```python
# PyTorch
import torch

# PyTorch does not need to input the value of low, which is equivalent to low=0 in MindSpore.
x = torch.randint(10, (3, 3))
print(tuple(x.shape))
# (3, 3)

# MindSpore
import mindspore

# MindSpore must take the default value of low in the torch (0 in this case) and pass it in as input.
x = mindspore.ops.randint(0, 10, (3, 3))
print(x.shape)
# (3, 3)
```
