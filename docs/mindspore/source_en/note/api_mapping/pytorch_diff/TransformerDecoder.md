# Differences between torch.nn.TransformerDecoder and mindspore.nn.TransformerDecoder

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerDecoder.md)

## torch.nn.TransformerDecoder

```python
class torch.nn.TransformerDecoder(
    decoder_layer,
    num_layers,
    norm=None
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

For more information, see [torch.nn.TransformerDecoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoder.html).

## mindspore.nn.TransformerDecoder

```python
class mindspore.nn.TransformerDecoder(
    decoder_layer,
    num_layers,
    norm=None
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

For more information, see [mindspore.nn.TransformerDecoder](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.TransformerDecoder.html).

## Differences

The usage of `mindspore.nn.TransformerDecoder` is mostly the same with that of `torch.nn.TransformerDecoder`.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | decoder_layer       | decoder_layer        | Consistent function |
|      | Parameter 2 | num_layers           | num_layers | Consistent function |
|      | Parameter 3 | norm        | norm | Consistent function |
| Input  | Input 1 | tgt            | tgt | Consistent function                                               |
|     | Input 2 | memory           | memory | Consistent function                                             |
|     | Input 3 | tgt_mask      | tgt_mask | In MindSpore, dtype can be set as float or Bool Tensor; in Pytorch dtype can be set as float, byte or Bool Tensor. |
|     | Input 4 | memory_mask      | memory_mask | In MindSpore, dtype can be set as float or Bool Tensor; in Pytorch dtype can be set as float, byte or Bool Tensor. |
|     | Input 5 | tgt_key_padding_mask      | tgt_key_padding_mask | In MindSpore, dtype can be set as float or Bool Tensor; in Pytorch dtype can be set as byte or Bool Tensor. |
|     | Input 6 | memory_key_padding_mask      | memory_key_padding_mask | In MindSpore, dtype can be set as float or Bool Tensor; in Pytorch dtype can be set as byte or Bool Tensor. |

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
