# 比较与torch.nn.TransformerEncoderLayer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/TransformerEncoderLayer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerEncoderLayer

```python
import torch
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
src = torch.rand(32, 10, 512)
out = encoder_layer(src)
```

更多内容详见[torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerEncoderLayer)。

## mindspore.parallel.nn.TransformerEncoderLayer

```python
class mindspore.parallel.nn.TransformerEncoderLayer(*args, **kwargs)(
    x, input_mask, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.parallel.nn.TransformerEncoderLayer](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.parallel.nn.html#mindspore.parallel.nn.TransformerEncoderLayer)。

## 使用方式

mindspore.parallel.nn.TransformerEncoderLayer在初始化参数和torch.nn.TransformerEncoderLayer并不完全相同，但是基本功能保持一致。具体的区别如下说明：

- mindspore.parallel.nn.TransformerEncoderLayer缺少src_key_padding_mask的输入。
- mindspore.parallel.nn.TransformerEncoderLayer提供了静态图的增量推理功能。
- mindspore.parallel.nn.TransformerEncoderLayer默认采用fp16进行矩阵运算。
- mindspore.parallel.nn.TransformerEncoderLayer的输入中encoder_mask是必须输入的。
- mindspore.parallel.nn.TransformerEncoderLayer会返回attention的key和value的计算结果。
- mindspore.parallel.nn.TransformerEncoderLayer提供了并行配置parallel_config入参，可以实现混合并行。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列长度等额外信息，并且在计算时需要输入encoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.parallel.nn import TransformerEncoderLayer
from mindspore import Tensor
model = TransformerEncoderLayer(batch_size=32, hidden_size=512,
                                ffn_hidden_size=2048, seq_length=10, num_heads=8)
encoder_input_value = Tensor(np.random.rand(32, 10, 512), mstype.float32)
encoder_input_mask = Tensor(np.ones((32, 10, 10)), mstype.float16)
output, past = model(encoder_input_value, encoder_input_mask)
# output:
# (32, 10, 512)

import torch
encoder_layer = torch.nn.TransformerEncoderLayerLayer(d_model=512, nhead=8)
transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
# Out：
# torch.Size([10, 32, 512])
```