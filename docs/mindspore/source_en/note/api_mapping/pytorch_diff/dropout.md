# Function Differences with torch.nn.Dropout

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

For more information, see [torch.nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html).

## mindspore.nn.Dropout

```python
mindspore.nn.Dropout(keep_prob=0.5, p=None)
```

For more information, see [mindspore.nn.Dropout](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Dropout.html).

## Differences

PyTorch: Dropout is a regularization device. The operator randomly sets some neuron outputs to 0 during training according to the dropout probability `p` , reducing overfitting by preventing correlation between neuron nodes.

MindSpore: MindSpore API implements much the same functionality as PyTorch. `keep_prob` is the input neuron retention rate, now deprecated, will be removed in the near future version.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
| Parameters | Parameter 1 | -   | keep_prob    | MindSpore discard parameter |
|      | Parameter 2 | p |  p   | The parameter names and functions are the same |
|      | Parameter 3 |   inplace   | - | MindSpore does not have this parameter |

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
output = torch.nn.Dropout(p=0.2, inplace=False)(input)
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
output = mindspore.nn.Dropout(p=0.2)(x)
print(output.shape)
# (5, 10)
```
