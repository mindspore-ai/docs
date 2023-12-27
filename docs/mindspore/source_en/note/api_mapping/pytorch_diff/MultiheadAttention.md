# Differences between torch.nn.MultiheadAttention and mindspore.nn.MultiheadAttention

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MultiheadAttention.md)

## torch.nn.MultiheadAttention

```python
class torch.nn.MultiheadAttention(
    embed_dim,
    num_heads,
    dropout=0.0,
    bias=True,
    add_bias_kv=False,
    add_zero_attn=False,
    kdim=None,
    vdim=None
)(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)
```

For more information, see [torch.nn.MultiheadAttention](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiheadAttention.html)。

## mindspore.nn.MultiheadAttention

```python
class mindspore.nn.MultiheadAttention(
    embed_dim,
    num_heads,
    dropout=0.0,
    has_bias=True,
    add_bias_kv=False,
    add_zero_attn=False,
    kdim=None,
    vdim=None,
    batch_first=False,
    dtype=mstype.float32
)(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True)
```

For more information, see [mindspore.nn.MultiheadAttention](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.MultiheadAttention.html)。

## Differences

The code implementation and parameter update logic of `mindspore.nn.MultiheadAttention` optimizer is mostly the same with `torch.nn.MultiheadAttention`.

| Categories  | Subcategories  | PyTorch                  | MindSpore     | Difference                                                 |
| ---- |-----  |------------------------- |-------------  |----------------------------------------------------|
| Parameters  | Parameter1 | embed_dim            | embed_dim | Consistent function                                               |
|      | Parameter2 | num_heads               | num_heads    | Consistent function                                        |
|      | Parameter3 | dropout               | dropout          | Consistent function           |
|      | Parameter4 | bias               | has_bias          | Consistent function           |
|      | Parameter5 | add_bias_kv               | add_bias_kv          | Consistent function           |
|      | Parameter6 | add_zero_attn        | add_zero_attn          | Consistent function |
|      | Parameter7 | kdim                 | kdim          | Consistent function |
|      | Parameter8 | vdim                 | vdim          | Consistent function |
|      | Parameter9 |                      | batch_first          | In MindSpore, first batch can be set as batch dimension, PyTorch does not have this function. |
|      | Parameter10 |                     | dtype          | In MindSpore, dtype can be set in Parameters using 'dtype'. PyTorch does not have this function. |
| Input  | Input1 | query            | query | Consistent function                                                |
|      | Input2 | key           | key | Consistent function                                                |
|      | Input3 | value      | value | Consistent function                                                |
|      | Input4 | key_padding_mask            | key_padding_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as byte or Bool Tensor. |
|      | Input5 | need_weights           | need_weights | Consistent function                                                |
|      | Input6 | attn_mask      | attn_mask | In MindSpore, dtype can be set as float or Bool Tensor; in PyTorch dtype can be set as float, byte or Bool Tensor. |
|      | Input7 |                | average_attn_weights | If true, indicates that the returned attn_weights should be averaged across heads. Otherwise, attn_weights are provided separately per head. PyTorch does not have this function. |

### Code Example

```python
# PyTorch
import torch
from torch import nn

embed_dim, num_heads = 128, 8
seq_length, batch_size = 10, 8
query = torch.rand(seq_length, batch_size, embed_dim)
key = torch.rand(seq_length, batch_size, embed_dim)
value = torch.rand(seq_length, batch_size, embed_dim)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
print(attn_output.shape)
print(attn_output_weights.shape)
#torch.Size([10, 8, 128])
#torch.Size([8, 10, 10])

# MindSpore
import mindspore as ms
import numpy as np

embed_dim, num_heads = 128, 8
seq_length, batch_size = 10, 8
query = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
key = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
value = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
multihead_attn = ms.nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
print(attn_output.shape)
#(10, 8, 128)
print(attn_output_weights.shape)
#(8, 10, 10)
```