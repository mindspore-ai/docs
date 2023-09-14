# 比较与torch.nn.TransformerDecoder的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerDecoder.md)

## torch.nn.TransformerDecoder

```python
class torch.nn.TransformerDecoder(
    decoder_layer,
    num_layers,
    norm=None
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

更多内容详见[torch.nn.TransformerDecoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoder.html)。

## mindspore.nn.TransformerDecoder

```python
class mindspore.nn.TransformerDecoder(
    decoder_layer,
    num_layers,
    norm=None
)(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
```

更多内容详见[mindspore.nn.TransformerDecoder](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TransformerDecoder.html)。

## 差异对比

`torch.nn.TransformerDecoder` 和 `mindspore.nn.TransformerDecoder` 用法基本一致。

| 分类  | 子类  | PyTorch                  | MindSpore     | 差异                                                 |
| ---- |-----  |------------------------- |-------------  |----------------------------------------------------|
| 参数  | 参数1 | decoder_layer            | decoder_layer | 功能一致                                               |
|      | 参数2 | num_layers               | num_layers    | 功能一致                                        |
|      | 参数3 | norm                     | norm          | 功能一致           |
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