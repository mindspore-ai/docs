# Function Differences with torch.randint

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/randint.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## torch.randint

```text
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

For more information, see [torch.randint](https://pytorch.org/docs/1.8.1/generated/torch.randint.html#torch.randint).

## mindspore.ops.randint

```text
mindspore.ops.randint(low, high, size, seed=None, *, dtype=None)
```

For more information, see [mindspore.ops.randint](https://www.mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.randint.html#mindspore.ops.randint).

## Differences

PyTorch：`low` is optional with default value: 0.

MindSpore：`low` is not optional and has no default value.

| Categories | Subcategories | PyTorch       | MindSpore | Differences                                               |
|------------|---------------|---------------|-----------|-----------------------------------------------------------|
| Parameters | Parameter 1   | low           | low       | `low` has default value 0 in PyTorch, MindSpore does not. |
|            | Parameter 2   | high          | high      | -                                                         |
|            | Parameter 3   | size          | size      | -                                                         |
|            | Parameter 4   | generator     | -         | General Differences                                       |
|            | Parameter 5   | out           | -         | General Differences                                       |
|            | Parameter 6   | dtype         | dtype     | -                                                         |
|            | Parameter 7   | layout        | -         | General Differences                                       |
|            | Parameter 8   | device        | -         | General Differences                                       |
|            | Parameter 9   | requires_grad | -         | General Differences                                       |
|            | Parameter 10  | -             | seed      | General Differences                                       |

### Code Example 1

```python
# PyTorch
import torch

# PyTorch does not need to set the value of low.
x = torch.randint(10, (3, 3))
print(tuple(x.shape))
# (3, 3)

# MindSpore
import mindspore

# MindSpore must set the default value of low in PyTorch(0 in this case), as one of the inputs.
x = mindspore.ops.randint(0, 10, (3, 3))
print(x.shape)
# (3, 3)
```
