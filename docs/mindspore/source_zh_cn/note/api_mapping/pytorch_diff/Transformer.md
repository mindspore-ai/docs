# 比较与torch.nn.Transformer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Transformer.md)

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

更多内容详见 [torch.nn.Transformer](https://pytorch.org/docs/1.8.1/generated/torch.nn.Transformer.html).

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

更多内容详见 [mindspore.nn.Transformer](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Transformer.html).

## 差异对比

The code implementation and parameter update logic of `mindspore.nn.Transformer` optimizer is mostly the same with `torch.nn.Transformer`.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| 参数 | 参数 1 | d_model       | d_model        | 功能一致 |
|      | 参数 2 | nhead           | nhead | 功能一致 |
|      | 参数 3 | num_encoder_layers        | num_encoder_layers | 功能一致 |
|      | 参数 4 | num_decoder_layers        | num_decoder_layers | 功能一致 |
|      | 参数 5 | dim_feedforward        | dim_feedforward | 功能一致 |
|      | 参数 6 | dropout        | dropout | 功能一致 |
|      | 参数 7 | activation        | activation | 功能一致 |
|      | 参数 8 | custom_encoder        | custom_encoder | 功能一致 |
|      | 参数 9 | custom_decoder        | custom_decoder | 功能一致 |
|      | 参数 10 |                | layer_norm_eps          | MindSpore可配置LayerNorm层的eps值, Pytorch没有此功能 |
|      | 参数 11 |                | batch_first          | MindSpore可配置第一维是否输出batch维度, Pytorch没有此功能 |
|      | 参数 12 |                | norm_first          | MindSpore可配置LayerNorm层是否位于MultiheadAttention层和FeedForward之间或之后, Pytorch没有此功能 |
|      | 参数 13 |                     | dtype          | MindSpore可配置网络参数的dtype， PyTorch没有此功能。 |
| 输入  | 输入 1 | src            | src | 功能一致                                              |
|     | 输入 2 | tgt            | tgt | 功能一致                                              |
|     | 输入 3 | src_mask           | src_mask | 功能一致                                            |
|     | 输入 4 | tgt_mask           | tgt_mask | 功能一致                                             |
|     | 输入 5 | memory_mask           | memory_mask | 功能一致                                             |
|     | 输入 6 | src_key_padding_mask      | src_key_padding_mask | 功能一致                       |
|     | 输入 7 | tgt_key_padding_mask      | tgt_key_padding_mask | 功能一致                     |
|     | 输入 8 | memory_key_padding_mask   | memory_key_padding_mask | 功能一致                      |

## 代码示例

```python
# PyTorch
import torch
from torch import nn

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand(10, 32, 512)
tgt = torch.rand(10, 32, 512)
out = transformer_model(src, tgt)

# MindSpore
import mindspore as ms
import numpy as np

transformer_model = ms.nn.Transformer(nhead=16, num_encoder_layers=12)
src = ms.Tensor(np.random.rand(10, 32, 512), ms.float32)
tgt = ms.Tensor(np.random.rand(20, 32, 512), ms.float32)
out = transformer_model(src, tgt)
```
