# Differences with torch.nn.Dropout

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Dropout.md)

## torch.nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

For more information, see [torch.nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html).

## mindspore.nn.Dropout

```python
mindspore.nn.Dropout(keep_prob=0.5, p=None, dtype=mstype.float32)
```

For more information, see [mindspore.nn.Dropout](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Dropout.html).

## Differences

PyTorch: Dropout is a regularization device. The operator randomly sets some neuron outputs to 0 during training according to the dropout probability `p` , reducing overfitting by preventing correlation between neuron nodes.

MindSpore: MindSpore API implements much the same functionality as PyTorch. `keep_prob` is the input neuron retention rate, now deprecated, will be removed in the near future version. `dtype` sets the data type of the output Tensor, now deprecated.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ----|
| Parameters | Parameter 1 | -   | keep_prob    | MindSpore discard parameter |
|      | Parameter 2 | p |  p   | The parameter names and functions are the same |
|      | Parameter 3 |   inplace   | - | MindSpore does not have this parameter |
|      | Parameter 4 |   -   | dtype | MindSpore discard parameter |

Dropout is often used to prevent training overfitting. It has an important probability value parameter. The meaning of this parameter in MindSpore is completely opposite to that in PyTorch and TensorFlow.

In MindSpore, the probability value corresponds to the `keep_prob` attribute of the Dropout operator, indicating the probability that the input is retained. `1-keep_prob` indicates the probability that the input is set to 0.

In PyTorch and TensorFlow, the probability values correspond to the attributes `p` and `rate` of the Dropout operator, respectively. They indicate the probability that the input is set to 0, which is opposite to the meaning of `keep_prob` in MindSpore.nn.Dropout.

In PyTorch, the network is in training mode by default, while in MindSpore, it's in inference mode by default. Therefore, by default, Dropout called by the network does not take effect and directly returns the input. Dropout can be performed only after the network is set to the training mode by using the `net.set_train()` method.

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
