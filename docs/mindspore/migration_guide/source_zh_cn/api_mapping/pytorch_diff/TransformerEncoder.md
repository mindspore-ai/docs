# 比较与torch.nn.TransformerEncoder的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/TransformerEncoder.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerEncoder

```python
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
```

更多内容详见[torch.nn.TransformerEncoder](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerEncoder)。

## mindspore.parallel.nn.TransformerEncoder

```python
class mindspore.parallel.nn.TransformerEncoder(*args, **kwargs)(
    hidden_states, attention_mask, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.parallel.nn.TransformerEncoder](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.parallel.nn.html#mindspore.parallel.nn.TransformerEncoder)。

## 使用方式

mindspore.parallel.nn.TransformerEncoder在初始化参数和torch.nn.TransformerEncoder并不完全相同，但是基本功能保持一致。torch.nn.TransformerEncoder采用了组合的方式，即将实例化的TransformerEncoderLayer作为torch.nn.TransformerEncoder的入参。而mindspore.nn.parallel.TransformerEncoder通过传入层的相关参数即可，跟TransformerEncoderLayer保持独立。具体的区别如下说明：

- mindspore.parallel.nn.TransformerEncoder缺少src_key_padding_mask的输入。
- mindspore.parallel.nn.TransformerEncoder提供了静态图的增量推理功能。
- mindspore.parallel.nn.TransformerEncoder默认采用fp16进行矩阵运算。
- mindspore.parallel.nn.TransformerEncoder的输入中encoder_mask是必须输入的。
- mindspore.parallel.nn.TransformerEncoder会返回以及encoder中每层attention的key和value的历史值。
- mindspore.parallel.nn.TransformerEncoder的初始化参数中缺少torch.nn.Transformer中的norm入参。
- mindspore.parallel.nn.TransformerEncoder提供了并行配置parallel_config入参，可以实现混合并行和流水线并行。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列长度等额外信息，并且在计算时需要输入encoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.parallel.nn import TransformerEncoder
from mindspore import Tensor
model = TransformerEncoder(batch_size=32, num_layers=2, hidden_size=512,
                           ffn_hidden_size=2048, seq_length=10, num_heads=8)
encoder_input_value = Tensor(np.random.rand(32, 10, 512), mstype.float32)
encoder_input_mask = Tensor(np.ones((32, 10, 10)), mstype.float16)
output, past = model(encoder_input_value, encoder_input_mask)
# output:
# (32, 10, 512)

import torch
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
# Out：
# torch.Size([10, 32, 512])
```