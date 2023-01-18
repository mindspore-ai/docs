# Function Differences with torch.nn.TransformerDecoderLayer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerDecoderLayer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.TransformerDecoderLayer

```python
torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

For more information, see [torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerDecoderLayer).

## mindspore.nn.transformer.TransformerDecoderLayer

```python
class mindspore.nn.transformer.TransformerDecoderLayer(hidden_size, ffn_hidden_size, num_heads, batch_size, src_seq_length, tgt_seq_length, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, use_past=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", moe_config=default_moe_config, parallel_config=default_dpmp_config)(
    hidden_stats, decoder_mask, encoder_output=None,
    memory_mask=None, init_reset=True, batch_valid_length=None
)
```

For more information, see [mindspore.nn.transformer.TransformerDecoderLayer](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerDecoderLayer).

## Usage

mindspore.nn.transformer.TransformerDecoderLayer is not exactly the same as torch.nn.TransformerDecoderLayer in terms of initialization parameters, but the basic functions remain the same. The specific differences are explained as follows.

| mindspore.nn.transformer.TransformerDecoderLayer | torch.nn.TransformerDecoderLayer | Descriptions                                                         |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| hidden_size                                   | d_model                          | The parameter names are inconsistent and have the same meaning.                                   |
| ffn_hidden_size                               | dim_feedforward                  | The parameter names are inconsistent and have the same meaning.                                   |
| num_heads                                     | nhead                            | The number of heads of Attention, same meaning.                              |
| batch_size                                    |                                  | MindSpore requires an additional batch size to be passed in for checksum and incremental inference.    |
| src_seq_length                                |                                  | encoder input sequence length.                                        |
| tgt_seq_length                                |                                  | decoder input sequence length.                                        |
| attention_dropout_rate                        | dropout                          | The meaning is different. attention_dropout_rate indicates the dropout at softmax, while PyTorch dropout parameter additionally controls the dropout rate of the hidden layer. |
| hidden_dropout_rate                           | dropout                          | The meaning is different. hidden_dropout_rate indicates the dropout at the hidden layer, while PyTorch dropout parameter additionally controls the dropout rate at the softmax. |
| post_layernorm_residual                       | norm_first                       | This parameter of MindSpore indicates whether to apply the layernorm to the input when summing the residuals, while PyTorch indicates whether to input the layernorm first when sub-layering the input.。 |
| use_past                                      |             | Whether to use incremental inference.                 |
| layernorm_compute_type                        |                                  | Control the calculation type of layernorm.                                    |
| softmax_compute_type                          |                                  | Control the calculation type of softmax in attention.                           |
| param_init_type                               |                                  | Control the type of parameter initialization.                                       |
| hidden_act                                    | activation                       | The type of the activation layer with the same meaning. MindSpore only supports strings.              |
| lambda_func                                   |                                  | See the API documentation for details on the configuration related to control parallelism.                            |
| moe_config                                    |                                  | Configuration parameters for MoE parallelism.                                          |
| parallel_config                               |                                  | Configuration parameters for parallel settings.                                         |
|                                               | layer_norm_eps                   | The value of the initial zero is prevented during the layernorm calculation.                              |
|                                               | batch_first                      | Whether batch is the 0th dimension in input and output Tensor. MindSpore takes the 0th dimension as the batch dimension, which corresponds to setting bathc_first=True in torch.nn.transformer. |

- mindspore.nn.transformer.TransformerDecoderLayer is missing input for tgt_key_padding_mask and emory_key_padding_mask.
- mindspore.nn.transformer.TransformerDecoderLayer provides incremental inference of static graphs.
- mindspore.nn.transformer.TransformerDecoderLayer uses fp16 for matrix operations by default.
- mindspore.nn.transformer.TransformerDecoderLayer input attention_mask is a mandatory input.
- mindspore.nn.transformer.TransformerDecoderLayer returns the history values of key and value for each layer of attention in encoder and decoder.
- mindspore.nn.transformer.TransformerDecoderLayer provides parallel configuration parallel_config input, which can realize mixed parallelism.

PyTorch: Instantiating the Transformer requires fewer parameters to be provided.

MindSpore: During class initialization, additional information such as batch_size, source and target sequence lengths need to be provided, and decoder_mask needs to be input during computation.

## Code Example

```python
import numpy as np
import mindspore as ms
from mindspore.nn.transformer import TransformerDecoderLayer
model = TransformerDecoderLayer(batch_size=32, hidden_size=512, ffn_hidden_size=2048,
                                num_heads=8, src_seq_length=10, tgt_seq_length=20)
encoder_input_value = ms.Tensor(np.ones((32, 10, 512)), ms.float32)
decoder_input_value = ms.Tensor(np.ones((32, 20, 512)), ms.float32)
decoder_input_mask = ms.Tensor(np.ones((32, 20, 20)), ms.float16)
memory_mask = ms.Tensor(np.ones((32, 20, 10)), ms.float16)
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
