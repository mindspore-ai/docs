# 比较与torch.nn.Transformer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Transformer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.Transformer

```python
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
```

更多内容详见[torch.nn.Transformer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Transformer)。

## mindspore.parallel.nn.Transformer

```python
class mindspore.parallel.nn.Transformer(*args, **kwargs)(
    encoder_inputs, encoder_masks, decoder_inputs=None,
    decoder_masks=None, memory_mask=None, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.parallel.nn.Transformer](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.parallel.nn.html#mindspore.parallel.nn.Transformer)。

## 使用方式

mindspore.parallel.nn.Transformer在初始化参数和torch.nn.Transformer并不完全相同，但是基本功能保持一致。是因为mindspore.nn.Transformer提供了更多细粒度的控制以及并行配置，可以轻松的实现并行训练。其中的主要区别概括如下：

- mindspore.parallel.nn.Transformer缺少src_key_padding_mask、tgt_key_padding_mask和memory_key_padding_mask输入。
- mindspore.parallel.nn.Transformer提供了静态图的增量推理功能。
- mindspore.parallel.nn.Transformer默认采用fp16进行矩阵运算。
- mindspore.parallel.nn.Transformer的输入中encoder_mask,decoder_mask是必须输入的。
- mindspore.parallel.nn.Transformer会返回decoder的输出值、以及encoder和decoder中每层attention的key,value的历史值。
- mindspore.parallel.nn.Transformer的输出值，是以batch为第0个维度的，对应于torch.nn.transformer中设置bathc_first=True。
- mindspore.parallel.nn.TransformerEncoder提供了并行配置parallel_config，可以实现混合并行和流水线并行。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列和目标序列的句子长度等额外信息，并且在计算时需要输入encoder_mask和decoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.parallel.nn import Transformer
from mindspore import Tensor
model = Transformer(batch_size=32, encoder_layers=1,
                    decoder_layers=1, hidden_size=512, ffn_hidden_size=2048,
                    src_seq_length=10, tgt_seq_length=20)
encoder_input_value = Tensor(np.random.rand(32, 10, 512), mstype.float32)
encoder_input_mask = Tensor(np.ones((32, 10, 10)), mstype.float16)
decoder_input_value = Tensor(np.random.rand(32, 20, 512), mstype.float32)
decoder_input_mask = Tensor(np.ones((32, 20, 20)), mstype.float16)
memory_mask = Tensor(np.ones((32, 20, 10)), mstype.float16)
output, en_past, de_past = model(encoder_input_value, encoder_input_mask, decoder_input_value,
                                 decoder_input_mask, memory_mask)
# output:
# (32, 20, 512)

import torch
transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12,
                                         num_encoder_layers=1, num_decoder_layers=1)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
# Out：
# torch.Size([20, 32, 512])
```