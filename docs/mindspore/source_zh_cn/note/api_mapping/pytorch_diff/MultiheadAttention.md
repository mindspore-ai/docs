# 比较与torch.nn.MultiheadAttention的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MultiheadAttention.md)

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
)
```

更多内容详见[torch.nn.MultiheadAttention](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiheadAttention.html)。

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
)
```

更多内容详见[mindspore.nn.MultiheadAttention](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MultiheadAttention.html)。

## 差异对比

`torch.nn.MultiheadAttention` 和 `mindspore.nn.MultiheadAttention` 用法基本一致。

| 分类  | 子类  | PyTorch                  | MindSpore     | 差异                                                 |
| ---- |-----  |------------------------- |-------------  |----------------------------------------------------|
| 参数  | 参数1  | embed_dim            | embed_dim | 功能一致                                               |
|      | 参数2  | num_heads               | num_heads    | 功能一致                                        |
|      | 参数3  | dropout               | dropout          | 功能一致           |
|      | 参数4  | has_bias               | has_bias          | 功能一致           |
|      | 参数5  | add_bias_kv               | add_bias_kv          | 功能一致           |
|      | 参数6  | add_zero_attn        | add_zero_attn          | 功能一致 |
|      | 参数7  | kdim                 | kdim          | 功能一致 |
|      | 参数8  | vdim                 | vdim          | 功能一致 |
|      | 参数9  |                      | batch_first          | MindSpore可配置第一维是否输出batch维度, PyTorch没有此功能。 |
|      | 参数10 |                     | dtype          | MindSpore可配置网络参数的dtype， PyTorch没有此功能。 |
| 输入  | 输入1  | query            | query | 功能一致                                               |
|      | 输入2  | key           | key | 功能一致                                               |
|      | 输入3  | value      | value | 功能一致                                               |
|      | 输入4  | key_padding_mask            | key_padding_mask | 功能一致                                               |
|      | 输入5  | need_weights           | need_weights | 功能一致                                               |
|      | 输入6  | attn_mask      | attn_mask | 功能一致                                               |
|      | 输入7  |                | average_attn_weights | 如果为 True， 则返回值 attn_output_weights 为注意力头的平均值。如果为 False，则 attn_weights 分别返回每个注意力头的值。PyTorch没有此功能。 |

### 代码示例

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

# MindSpore.
import mindspore as ms
import numpy as np

embed_dim, num_heads = 128, 8
seq_length, batch_size = 10, 8
query = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
key = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
value = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
multihead_attn = ms.nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
```