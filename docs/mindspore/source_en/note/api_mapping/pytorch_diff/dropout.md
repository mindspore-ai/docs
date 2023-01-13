# Function Differences with torch.nn.Dropout

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False) -> Tensor
```

For more information, see [torch.nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html?highlight=torch%20nn%20dropout#torch.nn.Dropout).

## mindspore.ops.dropout

```python
mindspore.ops.dropout(x, p=0.5, seed0=0, seed1=0) -> Tensor
```

For more information, see [mindspore.ops.dropout](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.dropout.html).

## Differences

PyTorch: dropout is a function used to prevent or mitigate overfitting by dropping a random portion of neurons at different training sessions. That is, the neuronal output is randomly set to 0 with a certain probability p, which serves to reduce the neuronal correlation. The remaining parameters that are not set to 0 will be scaled with $\frac{1}{1-p}$.

MindSpore: MindSpore API Basically achieves the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
| Parameters | Parameter 1 | p   | p    | -  |
|      | Parameter 2 | inplace |           | If set to True, this action will be performed in-place, the default value is False. In-place execution means that the operation is performed in the memory space of the input itself, i.e. Dropout is also performed on the input and save the input. MindSpore does not have this parameter. |
|      | Parameter 3 |         | x         | (Tensor), dropout input, Tensor of any dimension.      |
|      | Parameter 4 |         | seed0     | (int), the random seed of the operator layer, used to generate random numbers. Default value: 0     |
|      | Parameter 5 |         | seed1     | (int), the global random seed, which together with the random seed of the operator layer determines the finally generated random number. Default value: 0 |

### Code Example 1

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
from mindspore import ops
from mindspore import Tensor
x = Tensor([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]], mindspore.float32)
output, mask = ops.dropout(x, p=0.2)
print(output.shape)
# (5, 10)
```
