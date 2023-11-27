# 比较与torch.nn.TransformerDecoderLayer的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerDecoderLayer.md)

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

更多内容详见[torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoderLayer.html)。

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
    dtype=mstype.float32=False
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

更多内容详见[mindspore.nn.TransformerDecoderLayer](https://mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.TransformerDecoderLayer.html)。

## 差异对比

`torch.nn.TransformerDecoderLayer` 和 `mindspore.nn.TransformerDecoderLayer` 用法基本一致。

| 分类  | 子类  | PyTorch                  | MindSpore     | 差异                                                 |
| ---- |-----  |------------------------- |-------------  |----------------------------------------------------|
| 参数  | 参数1 | d_model            | d_model | 功能一致                                               |
|      | 参数2 | nhead               | nhead    | 功能一致                                        |
|      | 参数3 | dim_feedforward               | dim_feedforward          | 功能一致           |
|      | 参数4 | dropout               | dropout          | 功能一致           |
|      | 参数5 | activation               | activation          | 功能一致           |
|      | 参数6 |                | layer_norm_eps          | MindSpore可配置LayerNorm层的eps值, PyTorch没有此功能 |
|      | 参数7 |                | batch_first          | MindSpore可配置第一维是否输出batch维度, PyTorch没有此功能 |
|      | 参数8 |                | norm_first          | MindSpore可配置LayerNorm层是否位于Multiheadttention层和FeedForward之间或之后, PyTorch没有此功能 |
|      | 参数9 |                     | dtype          | MindSpore可配置网络参数的dtype， PyTorch没有此功能。 |
| 输入  | 输入1 | tgt            | tgt | 功能一致                                               |
|      | 输入2 | memory           | memory | 功能一致                                               |
|      | 输入3 | tgt_mask      | tgt_mask | MindSpore中dtype可设置为float或Bool Tensor，PyTorch中dtype可设置为float、byte或Bool Tensor |
|      | 输入4 | memory_mask            | memory_mask | MindSpore中dtype可设置为float或Bool Tensor，PyTorch中dtype可设置为float、byte或Bool Tensor |
|      | 输入5 | tgt_key_padding_mask           | tgt_key_padding_mask | MindSpore中dtype可设置为float或Bool Tensor，PyTorch中dtype可设置为byte或Bool Tensor |
|      | 输入6 | memory_key_padding_mask      | memory_key_padding_mask | MindSpore中dtype可设置为float或Bool Tensor，PyTorch中dtype可设置为byte或Bool Tensor |

### 代码示例

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