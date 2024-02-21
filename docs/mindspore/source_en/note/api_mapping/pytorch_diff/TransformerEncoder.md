# Differences between torch.nn.TransformerEncoder and mindspore.nn.TransformerEncoder

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerEncoder.md)

## torch.nn.TransformerEncoder

```python
class torch.nn.TransformerEncoder(
    encoder_layer,
    num_layers,
    norm=None
)(src, mask=None, src_key_padding_mask=None)
```

For more information, see [torch.nn.TransformerEncoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoder.html).

## mindspore.nn.TransformerEncoder

```python
class mindspore.nn.TransformerEncoder(
    encoder_layer,
    num_layers,
    norm=None
)(src, src_mask=None, src_key_padding_mask=None)
```

For more information, see [mindspore.nn.TransformerEncoder](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.TransformerEncoder.html).

## Differences

The usage of `mindspore.nn.TransformerEncoder` is mostly the same with that of `torch.nn.TransformerEncoder`.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | encoder_layer       | encoder_layer        | Consistent function |
|      | Parameter 2 | num_layers           | num_layers | Consistent function |
|      | Parameter 3 | norm        | norm | Consistent function |              |
| Input  | Input1 | src            | src | Consistent function                                               |
|     | Input2 | mask           | src_mask | Consistent function, different parameter names                                            |
|     | Input3 | src_key_padding_mask      | src_key_padding_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as byte or bool Tensor. |

### Code Example

```python
# PyTorch
import torch
from torch import nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
print(out.shape)
#torch.Size([10, 32, 512])

# MindSpore
import mindspore
from mindspore import nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = mindspore.numpy.rand(10, 32, 512)
out = transformer_encoder(src)
print(out.shape)
#(10, 32, 512)
```
