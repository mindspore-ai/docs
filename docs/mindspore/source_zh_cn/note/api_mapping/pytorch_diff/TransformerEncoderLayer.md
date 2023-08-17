# 比较与torch.nn.TransformerEncoderLayer的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerEncoderLayer.md)

## torch.nn.TransformerEncoderLayer

```python
class torch.nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu'
)
```

更多内容详见[torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoderLayer.html)。

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
    dtype=mstype.float32=False
)
```

更多内容详见[mindspore.nn.TransformerEncoderLayer](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TransformerEncoderLayer.html)。

## 差异对比

`torch.nn.TransformerEncoder` 和 `mindspore.nn.TransformerEncoder` 用法基本一致。

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
| 输入  | 输入1 | src            | src | 功能一致                                               |
|      | 输入2 | src_mask           | src_mask | 功能一致                                               |
|      | 输入3 | src_key_padding_mask      | src_key_padding_mask | 功能一致                                               |

### 代码示例

```python
# PyTorch
import torch
from torch import nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)

# MindSpore.
import mindspore
from mindspore import nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = mindspore.numpy.rand(10, 32, 512)
out = transformer_encoder(src)
```
