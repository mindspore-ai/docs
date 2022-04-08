# 比较与torch.nn.TransformerEncoderLayer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TransformerEncoderLayer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerEncoderLayer

```python
torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

更多内容详见[torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerEncoderLayer)。

## mindspore.nn.transformer.TransformerEncoderLayer

```python
class mindspore.nn.transformer.TransformerEncoderLayer(batch_size, hidden_size, ffn_hidden_size, num_heads, seq_length, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", use_past=False, moe_config=default_moe_config, parallel_config=default_dpmp_config)(
    x, input_mask, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.nn.transformer.TransformerEncoderLayer](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerEncoderLayer)。

## 使用方式

mindspore.nn.transformer.TransformerEncoderLayer在初始化参数和torch.nn.TransformerEncoderLayer并不完全相同，但是基本功能保持一致。具体的区别如下说明：

| mindspore.nn.transformer.TransformerEncoderLayer | torch.nn.TransformerEncoderLayer | 说明                                                         |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| batch_size                                    |                                  | MindSpore需要传入额外的batch size以作校验和增量推理使用。    |
| hidden_size                                   | d_model                          | 参数名称不一致，含义相同。                                   |
| seq_length                                    |                                  | encoder输入序列长度。                                        |
| ffn_hidden_size                               | dim_feedforward                  | 参数名称不一致，含义相同。                                   |
| num_heads                                     | nhead                            | Attention的head数目，含义相同。                              |
| hidden_dropout_rate                           | dropout                          | 含义不同。attention_dropout_rate表示在softmax处的dropout，而PyTorch的dropout参数额外控制了隐藏层的dropout rate。 |
| attention_dropout_rate                        | dropout                          | 含义不同。hidden_dropout_rate表示在隐藏层处的dropout，而PyTorch的dropout参数额外控制了softmax处的dropout rate。 |
| post_layernorm_residual                       | norm_first                       | 含义不同。MindSpore的该参数表示残差相加时对输入是否应用layernorm，而PyTorch表示输入子层时是否先输入layernorm。 |
| hidden_act                                    | activation                       | 激活层的类型，含义相同。MindSpore仅支持字符串。              |
| layernorm_compute_type                        |                                  | 控制layernorm的计算类型。                                    |
| softmax_compute_type                          |                                  | 控制attention中softmax的计算类型。                           |
| param_init_type                               |                                  | 控制参数初始化的类型。                                       |
| use_past                                      |                                  | 是否使用增量推理。                                           |
| moe_config                                    |                                  | MoE并行的配置参数。                                          |
| parallel_config                               |                                  | 并行设置的配置参数。                                         |

- mindspore.nn.transformer.TransformerEncoderLayer缺少src_key_padding_mask的输入。
- mindspore.nn.transformer.TransformerEncoderLayer提供了静态图的增量推理功能。
- mindspore.nn.transformer.TransformerEncoderLayer默认采用fp16进行矩阵运算。
- mindspore.nn.transformer.TransformerEncoderLayer的输入中encoder_mask是必须输入的。
- mindspore.nn.transformer.TransformerEncoderLayer会返回attention的key和value的计算结果。
- mindspore.nn.transformer.TransformerEncoderLayer提供了并行配置parallel_config入参，可以实现混合并行。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列长度等额外信息，并且在计算时需要输入encoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn.transformer import TransformerEncoderLayer
from mindspore import Tensor
model = TransformerEncoderLayer(batch_size=32, hidden_size=512,
                                ffn_hidden_size=2048, seq_length=10, num_heads=8)
encoder_input_value = Tensor(np.random.rand(32, 10, 512), mstype.float32)
encoder_input_mask = Tensor(np.ones((32, 10, 10)), mstype.float16)
output, past = model(encoder_input_value, encoder_input_mask)
print(output.shape)
# output:
# (32, 10, 512)

import torch
encoder_layer = torch.nn.TransformerEncoderLayerLayer(d_model=512, nhead=8)
transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
src = torch.rand(10, 32, 512)
output = transformer_encoder(src)
print(output.shape)
# output:
# torch.Size([10, 32, 512])
```