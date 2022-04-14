# 比较与torch.nn.TransformerDecoder的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerDecoder.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerDecoder

```python
torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
```

更多内容详见[torch.nn.TransformerDecoder](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerDecoder)。

## mindspore.nn.transformer.TransformerDecoder

```python
class mindspore.nn.transformer.TransformerDecoder(num_layers, batch_size, hidden_size, ffn_hidden_size, src_seq_length, tgt_seq_length, num_heads, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", lambda_func=None, use_past=False, offset=0, moe_config=default_moe_config, parallel_config=default_transformer_config)(
    hidden_stats, decoder_mask, encoder_output=None,
    memory_mask=None, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.nn.transformer.TransformerDecoder](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerDecoder)。

## 使用方式

mindspore.nn.transformer.TransformerDecoder在初始化参数和torch.nn.TransformerDecoder并不完全相同，但是基本功能保持一致。torch.nn.TransformerDecoder采用了组合的方式，即将实例化的TransformerDecoderLayer作为torch.nn.TransformerDecoder的入参。而mindspore.nn.parallel.TransformerDecoder通过传入层的相关参数即可，跟TransformerDecoderLayer保持独立。具体的区别如下说明：

| mindspore.nn.transformer.TransformerDecoder | torch.nn.TransformerDecoder | 说明                                                      |
| ---------------------------------------- | --------------------------- | --------------------------------------------------------- |
| num_layers                               | num_layers                  | 含义相同。                                                |
| batch_size                               |                             | MindSpore需要传入额外的batch size以作校验和增量推理使用。 |
| hidden_size                              |                             | 参数名称不一致，含义相同。                                |
| ffn_hidden_size                          |                             |                                                           |
| src_seq_length                           |                             | encoder输入序列长度。                                     |
| tgt_seq_length                           |                             | decoder输入序列长度。                                     |
| num_heads                                |                             |                                                           |
| attention_dropout_rate                   |                             | attention_dropout_rate表示在softmax处的dropout。          |
| hidden_dropout_rate                      |                             | hidden_dropout_rate表示在隐藏层处的dropout。              |
| post_layernorm_residual                  |                             | MindSpore的该参数表示残差相加时对输入是否应用layernorm。  |
| layernorm_compute_type                   |                             | 控制layernorm的计算类型。                                 |
| softmax_compute_type                     |                             | 控制attention中softmax的计算类型。                        |
| param_init_type                          |                             | 控制参数初始化的类型。                                    |
| hidden_act                               |                             | 激活层的类型，含义相同。MindSpore仅支持字符串。           |
| lambda_func                              |                             | 控制并行的相关配置，详见API文档。                         |
| use_past                                 |                             | 是否使用增量推理。                                        |
| offset                                   |                             | encoder用来计算fusion标记的初始值。                       |
| moe_config                               |                             | MoE并行的配置参数。                                       |
| parallel_config                          |                             | 并行设置的配置参数。                                      |
|                                          | decoder_layer               | decoder的实例化参数                                       |
|                                          | norm                        | 在decoder的输出是否应用传入的norm cell。                  |

- mindspore.nn.transformer.TransformerDecoder缺少tgt_key_padding_mask和emory_key_padding_mask的输入。
- mindspore.nn.transformer.TransformerDecoder提供了静态图的增量推理功能。
- mindspore.nn.transformer.TransformerDecoder默认采用fp16进行矩阵运算。
- mindspore.nn.transformer.TransformerDecoder的输入中attention_mask是必须输入的。
- mindspore.nn.transformer.TransformerDecoder会返回以及encoder、decoder中每层attention的key,value的历史值。
- mindspore.nn.transformer.TransformerDecoder的初始化参数中缺少torch.nn.Transformer中的norm入参。
- mindspore.nn.transformer.TransformerDecoder提供了并行配置parallel_config入参，可以实现混合并行和流水线并行。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列和目标序列长度等额外信息，并且在计算时需要输入decoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn.transformer import TransformerDecoderLayer
from mindspore import Tensor
model = TransformerDecoderLayer(batch_size=32, hidden_size=512, ffn_hidden_size=2048,
                                num_heads=8, src_seq_length=10, tgt_seq_length=20)
encoder_input_value = Tensor(np.ones((32, 10, 512)), mstype.float32)
decoder_input_value = Tensor(np.ones((32, 20, 512)), mstype.float32)
decoder_input_mask = Tensor(np.ones((32, 20, 20)), mstype.float16)
memory_mask = Tensor(np.ones((32, 20, 10)), mstype.float16)
output, past = model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)
print(output.shape)
# output:
# (32, 10, 512)

import torch
decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
output = transformer_decoder(tgt, memory)
print(output.shape)
# output
# torch.Size([10, 32, 512])
```