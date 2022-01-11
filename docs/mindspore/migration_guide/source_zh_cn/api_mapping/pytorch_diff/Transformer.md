# 比较与torch.nn.Transformer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Transformer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.nn.Transformer

```python
torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

更多内容详见[torch.nn.Transformer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Transformer)。

## mindspore.nn.transformer.Transformer

```python
class mindspore.nn.transformer.Transformer(hidden_size, batch_size, ffn_hidden_size, src_seq_length, tgt_seq_length, encoder_layers=3, decoder_layers=3, num_heads=2, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu", post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, lambda_func=None, use_past=False, moe_config=default_moe_config, parallel_config=default_transformer_config)(
    encoder_inputs, encoder_masks, decoder_inputs=None,
    decoder_masks=None, memory_mask=None, init_reset=True, batch_valid_length=None
)
```

更多内容详见[mindspore.nn.transformer.Transformer](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.Transformer)。

## 使用方式

mindspore.nn.transformer.Transformer在初始化参数和torch.nn.Transformer并不完全相同，但是基本功能保持一致。是因为mindspore.nn.Transformer提供了更多细粒度的控制以及并行配置，可以轻松的实现并行训练。其中的主要区别概括如下：

| mindspore.nn.transformer.Transformer | torch.nn.Transformer | 说明                                                         |
| --------------------------------- | -------------------- | ------------------------------------------------------------ |
| hidden_size                       | d_model              | 参数名称不一致，含义相同。                                   |
| batch_size                        |                      | MindSpore需要传入额外的batch size以作校验和增量推理使用。    |
| ffn_hidden_size                   | dim_feedforward      | 参数名称不一致，含义相同。                                   |
| src_seq_length                    |                      | encoder输入序列长度。                                        |
| tgt_seq_length                    |                      | decoder输入序列长度。                                        |
| encoder_layers                    | num_encoder_layers   | encoder的层数，含义相同。                                    |
| decoder_layers                    | num_decoder_layers   | decoder的层数，含义相同。                                    |
| num_heads                         | nhead                | Attention的head数目，含义相同。                              |
| attention_dropout_rate            | dropout              | 含义不同。attention_dropout_rate表示在softmax处的dropout，而PyTorch的dropout参数额外控制了隐藏层的dropout rate。 |
| hidden_dropout_rate               | dropout              | 含义不同。hidden_dropout_rate表示在隐藏层处的dropout，而PyTorch的dropout参数额外控制了softmax处的dropout rate。 |
| hidden_act                        | activation           | 激活层的类型，含义相同。MindSpore仅支持字符串。              |
| post_layernorm_residual           | norm_first           | 含义不同。MindSpore的该参数表示残差相加对输入是否应用layernorm，而PyTorch表示输入子层时是否先输入layernorm。 |
| layernorm_compute_type            |                      | 控制layernorm的计算类型。                                    |
| softmax_compute_type              |                      | 控制attention中softmax的计算类型。                           |
| param_init_type                   |                      | 控制参数初始化的类型。                                       |
| lambda_func                       |                      | 控制并行的相关配置，详见API文档。                            |
| use_past                          |                      | 是否使用增量推理。                                           |
| moe_config                        |                      | MoE并行的配置参数。                                          |
| parallel_config                   |                      | 并行设置的配置参数。                                         |
|                                   | custom_encoder       | 用户自定义的encoder。                                        |
|                                   | custom_decoder       | 用户自定义的decoder。                                        |
|                                   | layer_norm_eps       | layernorm计算时防止初零的数值。                              |
|                                   | batch_first          | 输入输出Tensor中batch是否为第0维度。MindSpore以第0个维度为batch维度，对应于torch.nn.transformer中设置bathc_first=True。 |

除了以上初始化参数不同之外，还有一些前向执行的输入和输出差异如下：

- mindspore.nn.transformer.Transformer缺少src_key_padding_mask、tgt_key_padding_mask和memory_key_padding_mask输入。

- mindspore.nn.transformer.Transformer的输入中encoder_mask,decoder_mask是必须输入的。

- mindspore.nn.transformer.Transformer会额外返回encoder和decoder中每层attention的key,value的历史值。

- mindspore.nn.transformer.Transformer中的post_layernorm_residual和torch.nn.transformer中的norm_first的参数对比如下：

  ```python
  # PyTorch
  if norm_fist:
      x = x + attention(norm(x))
  else:
      x = norm(x + attention(x))

  # MindSpore
  if post_layernorm_residual:
      x = norm(x) + attention(norm(x))
  else:
      x = x + attention(norm(x))
  ```

另外mindspore.nn.transformer.Transformer在功能上存在如下的差异：

- mindspore.nn.transformer.Transformer提供了静态图的增量推理功能。
- mindspore.nn.transformer.Transformer默认采用fp16进行矩阵运算。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列和目标序列的句子长度等额外信息，并且在计算时需要输入encoder_mask和decoder_mask。

## 代码示例

```python
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn.transformer import Transformer
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
print(output.shape)
# output:
# (32, 20, 512)

import torch
transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=1, num_decoder_layers=1)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
output = transformer_model(src, tgt)
print(output.shape)
# output:
# torch.Size([20, 32, 512])
```