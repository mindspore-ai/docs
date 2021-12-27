# 比较与torch.nn.TransformerEncoder的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/TransformerEncoder.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.TransformerEncoder

```python
torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
```

更多内容详见[torch.nn.TransformerEncoder](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerEncoder)。

## mindspore.nn.transformer.TransformerEncoder

```python
class mindspore.nn.transformer.TransformerEncoder(batch_size, num_layers, hidden_size, ffn_hidden_size, seq_length, num_heads, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu", post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, lambda_func=None, offset=0, use_past=False, moe_config=default_moe_config, parallel_config=default_transformer_config)(
    hidden_states, attention_mask, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.nn.transformer.TransformerEncoder](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerEncoder)。

## 使用方式

mindspore.nn.transformer.TransformerEncoder在初始化参数和torch.nn.TransformerEncoder并不完全相同，但是基本功能保持一致。torch.nn.TransformerEncoder采用了组合的方式，即将实例化的TransformerEncoderLayer作为torch.nn.TransformerEncoder的入参。而mindspore.nn.parallel.TransformerEncoder通过传入层的相关参数即可，跟TransformerEncoderLayer保持独立。具体的区别如下说明：

| mindspore.nn.transformer.TransformerEncoder | torch.nn.TransformerEncoder | 说明                                                      |
| ---------------------------------------- | --------------------------- | --------------------------------------------------------- |
| batch_size                               |                             | MindSpore需要传入额外的batch size以作校验和增量推理使用。 |
| num_layers                               | num_layers                  | 含义相同。                                                |
| hidden_size                              | d_model                     | 参数名称不一致，含义相同。                                |
| ffn_hidden_size                          |                             |                                                           |
| seq_length                               |                             | encoder输入序列长度。                                     |
| num_heads                                |                             |                                                           |
| hidden_dropout_rate                      |                             | hidden_dropout_rate表示在隐藏层处的dropout。              |
| attention_dropout_rate                   |                             | attention_dropout_rate表示在softmax处的dropout。          |
| post_layernorm_residual                  |                             | MindSpore的该参数表示残差相加时对输入是否应用layernorm。  |
| hidden_act                               | activation                  | 激活层的类型，含义相同。MindSpore仅支持字符串。           |
| layernorm_compute_type                   |                             | 控制layernorm的计算类型。                                 |
| softmax_compute_type                     |                             | 控制attention中softmax的计算类型。                        |
| param_init_type                          |                             | 控制参数初始化的类型。                                    |
| use_past                                 |                             | 是否使用增量推理。                                        |
| lambda_func                              |                             | 控制并行的相关配置，详见API文档。                         |
| offset                                   |                             | encoder用来计算fusion标记的初始值。                       |
| moe_config                               |                             | MoE并行的配置参数。                                       |
| parallel_config                          |                             | 并行设置的配置参数。                                      |
|                                          | norm                        | 在encoder的输出是否应用传入的norm cell。                  |

- mindspore.nn.transformer.TransformerEncoder缺少src_key_padding_mask的输入。
- mindspore.nn.transformer.TransformerEncoder提供了静态图的增量推理功能。
- mindspore.nn.transformer.TransformerEncoder默认采用fp16进行矩阵运算。
- mindspore.nn.transformer.TransformerEncoder的输入中encoder_mask是必须输入的。
- mindspore.nn.transformer.TransformerEncoder会返回以及encoder中每层attention的key和value的历史值。
- mindspore.nn.transformer.TransformerEncoder的初始化参数中缺少torch.nn.Transformer中的norm入参。
- mindspore.nn.transformer.TransformerEncoder提供了并行配置parallel_config入参，可以实现混合并行和流水线并行。

PyTorch：实例化TransformerEncoder时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列长度等额外信息，并且在计算时需要输入encoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn.transformer import TransformerEncoder
from mindspore import Tensor
model = TransformerEncoder(batch_size=32, num_layers=2, hidden_size=512,
                           ffn_hidden_size=2048, seq_length=10, num_heads=8)
encoder_input_value = Tensor(np.random.rand(32, 10, 512), mstype.float32)
encoder_input_mask = Tensor(np.ones((32, 10, 10)), mstype.float16)
output, past = model(encoder_input_value, encoder_input_mask)
print(output.shape)
# output:
# (32, 10, 512)

import torch
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
src = torch.rand(10, 32, 512)
output = transformer_encoder(src)
print(output.shape)
# output:
# torch.Size([10, 32, 512])
```