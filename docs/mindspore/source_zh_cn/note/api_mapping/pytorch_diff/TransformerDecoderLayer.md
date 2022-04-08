# 比较与torch.nn.TransformerDecoderLayer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerDecoderLayer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerDecoderLayer

```python
torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

更多内容详见[torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerDecoderLayer)。

## mindspore.nn.transformer.TransformerDecoderLayer

```python
class mindspore.nn.transformer.TransformerDecoderLayer(hidden_size, ffn_hidden_size, num_heads, batch_size, src_seq_length, tgt_seq_length, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, use_past=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", moe_config=default_moe_config, parallel_config=default_dpmp_config)(
    hidden_stats, decoder_mask, encoder_output=None,
    memory_mask=None, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.nn.transformer.TransformerDecoderLayer](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerDecoderLayer)。

## 使用方式

mindspore.nn.transformer.TransformerDecoderLayer在初始化参数和torch.nn.TransformerDecoderLayer并不完全相同，但是基本功能保持一致。具体的区别如下说明：

| mindspore.nn.transformer.TransformerDecoderLayer | torch.nn.TransformerDecoderLayer | 说明                                                         |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| hidden_size                                   | d_model                          | 参数名称不一致，含义相同。                                   |
| ffn_hidden_size                               | dim_feedforward                  | 参数名称不一致，含义相同。                                   |
| num_heads                                     | nhead                            | Attention的head数目，含义相同。                              |
| batch_size                                    |                                  | MindSpore需要传入额外的batch size以作校验和增量推理使用。    |
| src_seq_length                                |                                  | encoder输入序列长度。                                        |
| tgt_seq_length                                |                                  | decoder输入序列长度。                                        |
| attention_dropout_rate                        | dropout                          | 含义不同。attention_dropout_rate表示在softmax处的dropout，而PyTorch的dropout参数额外控制了隐藏层的dropout rate。 |
| hidden_dropout_rate                           | dropout                          | 含义不同。hidden_dropout_rate表示在隐藏层处的dropout，而PyTorch的dropout参数额外控制了softmax处的dropout rate。 |
| post_layernorm_residual                       | norm_first                       | 含义不同。MindSpore的该参数表示残差相加时对输入是否应用layernorm，而PyTorch表示输入子层时是否先输入layernorm。 |
| use_past                                      |                                  | 是否使用增量推理。                                           |
| layernorm_compute_type                        |                                  | 控制layernorm的计算类型。                                    |
| softmax_compute_type                          |                                  | 控制attention中softmax的计算类型。                           |
| param_init_type                               |                                  | 控制参数初始化的类型。                                       |
| hidden_act                                    | activation                       | 激活层的类型，含义相同。MindSpore仅支持字符串。              |
| lambda_func                                   |                                  | 控制并行的相关配置，详见API文档。                            |
| moe_config                                    |                                  | MoE并行的配置参数。                                          |
| parallel_config                               |                                  | 并行设置的配置参数。                                         |
|                                               | layer_norm_eps                   | layernorm计算时防止初零的数值。                              |
|                                               | batch_first                      | 输入输出Tensor中batch是否为第0维度。MindSpore以第0个维度为batch维度，对应于torch.nn.TransformerDecoderLayer中设置bathc_first=True。 |

- mindspore.nn.transformer.TransformerDecoderLayer缺少tgt_key_padding_mask和emory_key_padding_mask的输入。
- mindspore.nn.transformer.TransformerDecoderLayer提供了静态图的增量推理功能。
- mindspore.nn.transformer.TransformerDecoderLayer默认采用fp16进行矩阵运算。
- mindspore.nn.transformer.TransformerDecoderLayer的输入中attention_mask是必须输入的。
- mindspore.nn.transformer.TransformerDecoderLayer会返回以及encoder, decoder中每层attention的key,value的历史值。
- mindspore.nn.transformer.TransformerDecoderLayer提供了并行配置parallel_config入参，可以实现混合并行。

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
# (32, 20, 512)

import torch
decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
output = decoder_layer(tgt, memory)
print(output.shape)
# output:
# torch.Size([20, 32, 512])
```