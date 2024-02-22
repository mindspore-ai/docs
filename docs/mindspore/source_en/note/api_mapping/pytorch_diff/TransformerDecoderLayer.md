# Differences between torch.nn.TransformerDecoderLayer and mindspore.nn.TransformerDecoderLayer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerDecoderLayer.md)

## torch.nn.TransformerDecoderLayer

```python
class torch.nn.TransformerDecoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu'
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

For more information, see [torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoderLayer.html).

## mindspore.nn.TransformerDecoderLayer

```python
class mindspore.nn.TransformerDecoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False,
    dtype=mstype.float32
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

For more information, see [mindspore.nn.TransformerDecoderLayer](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.TransformerDecoderLayer.html).

## Differences

The code implementation and parameter update logic of `mindspore.nn.TransformerDecoderLayer` optimizer is mostly the same with `torch.nn.TransformerDecoderLayer`.

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
| Input  | Input 1 | tgt            | tgt | Consistent function                                               |
|     | Input 2 | memory           | memory | Consistent function                                             |
|     | Input 3 | tgt_mask      | tgt_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as float, byte or bool Tensor. |
|     | Input 4 | memory_mask      | memory_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as float, byte or bool Tensor. |
|     | Input 5 | tgt_key_padding_mask      | tgt_key_padding_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as byte or bool Tensor. |
|     | Input 6 | memory_key_padding_mask      | memory_key_padding_mask | In MindSpore, dtype can be set as float or bool Tensor; in PyTorch dtype can be set as byte or bool Tensor. |

### Code Example

```python
# PyTorch
import torch
from torch import nn

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
print(out.shape)
#torch.Size([20, 32, 512])

# MindSpore
import mindspore as ms
import numpy as np

decoder_layer = ms.nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = ms.nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = ms.Tensor(np.random.rand(10, 32, 512), ms.float32)
tgt = ms.Tensor(np.random.rand(20, 32, 512), ms.float32)
out = transformer_decoder(tgt, memory)
print(out.shape)
#(20, 32, 512)
```
