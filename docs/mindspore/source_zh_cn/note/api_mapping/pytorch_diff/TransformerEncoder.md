# 比较与torch.nn.TransformerEncoder的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerEncoder.md)

## torch.nn.TransformerEncoder

```python
class torch.nn.TransformerEncoder(
    encoder_layer,
    num_layers,
    norm=None
)(src, mask, src_key_padding_mask)
```

更多内容详见[torch.nn.TransformerEncoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoder.html)。

## mindspore.nn.TransformerEncoder

```python
class mindspore.nn.TransformerEncoder(
    encoder_layer,
    num_layers,
    norm=None
)(src, src_mask, src_key_padding_mask)
```

更多内容详见[mindspore.nn.TransformerEncoder](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.TransformerEncoder.html)。

## 差异对比

`torch.nn.TransformerEncoder` 和 `mindspore.nn.TransformerEncoder` 用法基本一致。

| 分类  | 子类  | PyTorch                  | MindSpore     | 差异                                                 |
| ---- |-----  |------------------------- |-------------  |----------------------------------------------------|
| 参数  | 参数1 | encoder_layer            | encoder_layer | 功能一致                                               |
|      | 参数2 | num_layers               | num_layers    | 功能一致                                        |
|      | 参数3 | norm                     | norm          | 功能一致                                         |
| 输入  | 输入1 | src            | src | 功能一致                                               |
|     | 输入2 | mask           | src_mask | 功能一致，参数名不同                                               |
|     | 输入3 | src_key_padding_mask      | src_key_padding_mask | 功能一致                                               |

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
