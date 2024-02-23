# Differences between torch.nn.TransformerEncoderLayer and mindspore.nn.TransformerEncoderLayer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerEncoderLayer.md)

## torch.nn.TransformerEncoderLayer

```python
class torch.nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu'
)(src, src_mask=None, src_key_padding_mask=None)
```

For more information, see [torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoderLayer.html).

## mindspore.nn.TransformerEncoderLayer

```python
class mindspore.nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False,
    dtype=mstype.float32
)(src, src_mask=None, src_key_padding_mask=None)
```

For more information, see [mindspore.nn.TransformerEncoderLayer](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TransformerEncoderLayer.html).

## Differences

The usage of `mindspore.nn.TransformerEncoderLayer` is mostly the same with that of `torch.nn.TransformerEncoderLayer`.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | d_model       | d_model        | Consistent function |
|      | Parameter 2 | nhead           | nhead | Consistent function |
|      | Parameter 3 | dim_feedforward        | dim_feedforward | Consistent function |
|      | Parameter 4 | dropout        | dropout | Consistent function |
|      | Parameter 5 | activation        | activation | Consistent function |
|      | Parameter 6 |         | layer_norm_eps | In MindSpore, the value of eps can be set in LayerNorm, PyTorch does not have this function |
|      | Parameter 7 |         | batch_first | In MindSpore, first batch can be set as batch dimension, PyTorch does not have this function |
|      | Parameter 8 |         | norm_first | In MindSpore, LayerNorm can be set in between Multiheadttention Layer and FeedForward Layer or after, PyTorch does not have this function |
|      | Parameter 9 |         | dtype          | In MindSpore, dtype can be set in Parameters using 'dtype'. PyTorch does not have this function. |
| Input  | Input 1 | src            | src | Consistent function                                               |
|     | Input 2 | src_mask           | src_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as float, byte or bool Tensor. |
|     | Input 3 | src_key_padding_mask      | src_key_padding_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as byte or bool Tensor. |

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
