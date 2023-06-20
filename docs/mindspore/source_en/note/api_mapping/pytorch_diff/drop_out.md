# Function differences with torch.nn.functional.dropout

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/drop_out.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.dropout

```python
torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
```

For more information, see [torch.nn.functional.dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.dropout).

## mindspore.ops.dropout

```python
mindspore.ops.dropout(input, p=0.5, training=True, seed=None)
```

For more information, see [mindspore.ops.dropout](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.dropout.html).

## Differences

MindSpore API implements basically the same functions as PyTorch, but due to the different framework mechanisms, the input differences are as follows:

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 |    input  | input  | Consistent  |
|      | Parameter 2 |    p     |  p     | Consistent  |
|      | Parameter 3 | training  |  training | Consistent  |
|      | Parameter 4 | inplace |  -  | MindSpore does not have this parameter |
|      | Parameter 5 |    -    |  seed  | The seed of the random number generator. PyTorch does not have this parameter |

### Code Example

> When the inplace input is False, both APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor
input = tensor([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]])
output = torch.nn.functional.dropout(input)
print(output.shape)
# torch.Size([5, 10])

# MindSpore
import mindspore
from mindspore import Tensor
x = Tensor([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]], mindspore.float32)
output = mindspore.ops.dropout(x)
print(output.shape)
# (5, 10)
```
