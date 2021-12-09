# 比较与torch.nn.TransformerDecoder的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/TransformerDecoder.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerDecoder

```python
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
```

更多内容详见[torch.nn.TransformerDecoder](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerDecoder)。

## mindspore.parallel.nn.TransformerDecoder

```python
class mindspore.parallel.nn.TransformerDecoder(*args, **kwargs)(
    hidden_stats, decoder_mask, encoder_output=None,
    memory_mask=None, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.parallel.nn.TransformerDecoder](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.parallel.nn.html#mindspore.parallel.nn.TransformerDecoder)。

## 使用方式

mindspore.parallel.nn.TransformerDecoder在初始化参数和torch.nn.TransformerDecoder并不完全相同，但是基本功能保持一致。torch.nn.TransformerDecoder采用了组合的方式，即将实例化的TransformerDecoderLayer作为torch.nn.TransformerDecoder的入参。而mindspore.nn.parallel.TransformerDecoder通过传入层的相关参数即可，跟TransformerDecoderLayer保持独立。具体的区别如下说明：

- mindspore.parallel.nn.TransformerDecoder缺少tgt_key_padding_mask和emory_key_padding_mask的输入。
- mindspore.parallel.nn.TransformerDecoder提供了静态图的增量推理功能。
- mindspore.parallel.nn.TransformerDecoder默认采用fp16进行矩阵运算。
- mindspore.parallel.nn.TransformerDecoder的输入中attention_mask是必须输入的。
- mindspore.parallel.nn.TransformerDecoder会返回以及encoder、decoder中每层attention的key,value的历史值。
- mindspore.parallel.nn.TransformerDecoder的初始化参数中缺少torch.nn.Transformer中的norm入参。
- mindspore.parallel.nn.TransformerDecoder提供了并行配置parallel_config入参，可以实现混合并行和流水线并行。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列和目标序列长度等额外信息，并且在计算时需要输入decoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.parallel.nn import TransformerDecoderLayer
from mindspore import Tensor
model = TransformerDecoderLayer(batch_size=32, hidden_size=512, ffn_hidden_size=2048,
                                num_heads=8, src_seq_length=10, tgt_seq_length=20)
encoder_input_value = Tensor(np.ones((32, 10, 512)), mstype.float32)
decoder_input_value = Tensor(np.ones((32, 20, 512)), mstype.float32)
decoder_input_mask = Tensor(np.ones((32, 20, 20)), mstype.float16)
memory_mask = Tensor(np.ones((32, 20, 10)), mstype.float16)
output, past = model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)
# output:
# (32, 10, 512)

import torch
decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
# Out：
# torch.Size([10, 32, 512])
```