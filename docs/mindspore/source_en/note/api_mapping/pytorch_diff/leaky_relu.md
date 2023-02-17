# Function Differences with torch.nn.functional.leaky_relu

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/leaky_relu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.leaky_relu

```text
torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor
```

For more information, see [torch.nn.functional.leaky_relu](https://pytorch.org/docs/1.8.1/nn.functional.html#leaky-relu).

## mindspore.ops.leaky_relu

```text
mindspore.ops.leaky_relu(x, alpha=0.2) -> Tensor
```

For more information, see [mindspore.ops.leaky_relu](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.leaky_relu.html#mindspore.ops.leaky_relu).

## Differnnces

PyTorch: The leaky_relu activation function. Elements that are less than 0 in `input` are multiplied by `negative_slope`.

MindSpore: MindSpore API basically implements the same function as PyTorch. The difference is that the initial value of `alpha` in MindSpore is 0.2, while the corresponding `negative_slope` in PyTorch has an initial value of 0.01.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | input | x  | Same function, different parameter names          |
|      | Parameter 2 | negative_slope | alpha | Same function, different parameter names |
|      | Parameter 3 | inplace | -     | Whether to make in-place changes to parameters. MindSpore does not have this feature |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
output = torch.nn.functional.leaky_relu(input, negative_slope=0.5, inplace=False)
print(output)
# tensor([-1.0000, -0.5000,  0.0000,  1.0000,  2.0000])

# MindSpore
import mindspore

input = mindspore.Tensor([-2, -1, 0, 1, 2], dtype=mindspore.float32)
output = mindspore.ops.leaky_relu(input, alpha=0.5)
print(output)
# [-1.  -0.5  0.   1.   2. ]
```

