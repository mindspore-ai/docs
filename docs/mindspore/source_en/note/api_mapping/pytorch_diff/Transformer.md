# Differences between torch.nn.Transformer and mindspore.nn.Transformer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Transformer.md)

## torch.nn.Transformer

```python
class torch.nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    custom_encoder=None,
    custom_decoder=None
)(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

For more information, see [torch.nn.Transformer](https://pytorch.org/docs/1.8.1/generated/torch.nn.Transformer.html).

## mindspore.nn.Transformer

```python
class mindspore.nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    custom_encoder=None,
    custom_decoder=None,
    layer_norm_eps=1e-05,
    batch_first=False,
    norm_first=False,
    dtype=mstype.float32
)(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

For more information, see [mindspore.nn.Transformer](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Transformer.html).

## Differences

The code implementation and parameter update logic of `mindspore.nn.Transformer` optimizer is mostly the same with `torch.nn.Transformer`.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | d_model       | d_model        | Consistent function |
|      | Parameter 2 | nhead           | nhead | Consistent function |
|      | Parameter 3 | num_encoder_layers        | num_encoder_layers | Consistent function |
|      | Parameter 4 | num_decoder_layers        | num_decoder_layers | Consistent function |
|      | Parameter 5 | dim_feedforward        | dim_feedforward | Consistent function |
|      | Parameter 6 | dropout        | dropout | Consistent function |
|      | Parameter 7 | activation        | activation | Consistent function |
|      | Parameter 8 | custom_encoder        | custom_encoder | Consistent function |
|      | Parameter 9 | custom_decoder        | custom_decoder | Consistent function |
|      | Parameter 10 |         | layer_norm_eps | In MindSpore, the value of eps can be set in LayerNorm, PyTorch does not have this function |
|      | Parameter 11 |         | batch_first | In MindSpore, first batch can be set as batch dimension, PyTorch does not have this function |
|      | Parameter 12 |         | norm_first | In MindSpore, LayerNorm can be set in between MultiheadAttention Layer and FeedForward Layer or after, PyTorch does not have this function |
|      | Parameter 13 |         | dtype          | In MindSpore, dtype can be set for parameters using 'dtype'. PyTorch does not have this function. |
| Input  | Input 1 | src            | src | Consistent function                                               |
|     | Input 2 | tgt            | tgt | Consistent function                                               |
|     | Input 3 | src_mask           | src_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as float, byte or Bool Tensor. |
|     | Input 4 | tgt_mask           | tgt_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as float, byte or Bool Tensor. |
|     | Input 5 | memory_mask           | memory_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as float, byte or Bool Tensor. |
|     | Input 6 | src_key_padding_mask      | src_key_padding_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as byte or Bool Tensor. |
|     | Input 7 | tgt_key_padding_mask      | tgt_key_padding_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as byte or Bool Tensor. |
|     | Input 8 | memory_key_padding_mask   | memory_key_padding_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as byte or Bool Tensor. |

## Code Example

```python
# PyTorch
import torch
from torch import nn

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand(10, 32, 512)
tgt = torch.rand(10, 32, 512)
out = transformer_model(src, tgt)
print(out.shape)
#torch.Size([10, 32, 512])

# MindSpore
import mindspore as ms
import numpy as np

transformer_model = ms.nn.Transformer(nhead=16, num_encoder_layers=12)
src = ms.Tensor(np.random.rand(10, 32, 512), ms.float32)
tgt = ms.Tensor(np.random.rand(10, 32, 512), ms.float32)
out = transformer_model(src, tgt)
print(out.shape)
#(10, 32, 512)
```
