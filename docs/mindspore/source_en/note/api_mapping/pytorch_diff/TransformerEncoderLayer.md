# Function Differences with torch.nn.TransformerEncoderLayer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerEncoderLayer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.TransformerEncoderLayer

```python
torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

For more information, see [torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerEncoderLayer).

## mindspore.nn.transformer.TransformerEncoderLayer

```python
class mindspore.nn.transformer.TransformerEncoderLayer(batch_size, hidden_size, ffn_hidden_size, num_heads, seq_length, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", use_past=False, moe_config=default_moe_config, parallel_config=default_dpmp_config)(
    x, input_mask, init_reset=True, batch_valid_length=None
)
```

For more information, see [mindspore.nn.transformer.TransformerEncoderLayer](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerEncoderLayer).

## Usage

mindspore.nn.transformer.TransformerEncoderLayer is not exactly the same as torch.nn.TransformerEncoderLayer in terms of initialization parameters, but the basic function remains the same. The specific differences are described below:

| mindspore.nn.transformer.TransformerEncoderLayer | torch.nn.TransformerEncoderLayer | Descriptions                                                         |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| batch_size                                    |                                  | MindSpore需要传入额外的batch size以作校验和增量推理使用。    |
| hidden_size                                   | d_model                          | The parameter names are inconsistent and have the same meaning.                                    |
| seq_length                                    |                                  | encoder input sequence length.                                        |
| ffn_hidden_size                               | dim_feedforward                  | The parameter names are inconsistent and have the same meaning.                                    |
| num_heads                                     | nhead                            | The number of heads of Attention, same meaning.                              |
| hidden_dropout_rate                           | dropout                          | The meaning is different. attention_dropout_rate indicates the dropout at softmax, while PyTorch dropout parameter additionally controls the dropout rate of the hidden layer. |
| attention_dropout_rate                        | dropout                          | The meaning is different. hidden_dropout_rate indicates the dropout at the hidden layer, while PyTorch dropout parameter additionally controls the dropout rate at the softmax. |
| post_layernorm_residual                       | norm_first                       | The meaning is different. MindSpore parameter indicates whether to apply the layernorm to the input when summing residuals, while PyTorch indicates whether to input the layernorm first when inputting sublayers. |
| hidden_act                                    | activation                       | The type of the activation layer with the same meaning. MindSpore only supports strings.               |
| layernorm_compute_type                        |                                  | Controls the calculation type of layernorm.                                    |
| softmax_compute_type                          |                                  | Control the calculation type of softmax in attention.                            |
| param_init_type                               |                                  | Controls the type of parameter initialization.                                   |
| use_past                                      |                                  | Whether to use incremental inference.                                           |
| moe_config                                    |                                  | Configuration parameters for MoE parallelism.                                           |
| parallel_config                               |                                  | Configuration parameters for parallel settings.                                         |

- mindspore.nn.transformer.TransformerEncoderLayer is missing input for src_key_padding_mask.
- mindspore.nn.transformer.TransformerEncoderLayer provides incremental inference functions for static graphs.
- mindspore.nn.transformer.TransformerEncoderLayer uses fp16 for matrix operations by default.
- encoder_mask in mindspore.nn.transformer.TransformerEncoder input is mandatory.
- mindspore.nn.transformer.TransformerEncoderLayer returns the result of key and value computation of attention.
- mindspore.nn.transformer.TransformerEncoderLayer provides parallel configuration parallel_config inputs to enable mixed parallelism.

PyTorch: Instantiating the TransformerEncoder requires fewer parameters to be provided.

MindSpore: During class initialization, additional information such as batch_size, source sequence length, etc. needs to be provided, and encoder_mask needs to be input during computation.

## Code Example

```python
import numpy as np
import mindspore as ms
from mindspore.nn.transformer import TransformerEncoderLayer
model = TransformerEncoderLayer(batch_size=32, hidden_size=512,
                                ffn_hidden_size=2048, seq_length=10, num_heads=8)
encoder_input_value = ms.Tensor(np.random.rand(32, 10, 512), ms.float32)
encoder_input_mask = ms.Tensor(np.ones((32, 10, 10)), ms.float16)
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
