# Function Differences with torch.nn.TransformerEncoder

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerEncoder.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.TransformerEncoder

```python
torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
```

For more information, see [torch.nn.TransformerEncoder](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerEncoder).

## mindspore.nn.transformer.TransformerEncoder

```python
class mindspore.nn.transformer.TransformerEncoder(batch_size, num_layers, hidden_size, ffn_hidden_size, seq_length, num_heads, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu", post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, lambda_func=None, offset=0, use_past=False, moe_config=default_moe_config, parallel_config=default_transformer_config)(
    hidden_states, attention_mask, init_reset=True, batch_valid_length=None
)
```

For more information, see [mindspore.nn.transformer.TransformerEncoder](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerEncoder).

## Usage

The mindspore.nn.transformer.TransformerEncoder does not have exactly the same initialization parameters as torch.nn.TransformerEncoder, but the basic function remains the same. torch.nn.TransformerEncoder uses a combination of TransformerEncoder, i.e. the instantiated TransformerEncoderLayer is used as an input to torch.nn.TransformerEncoder. TransformerEncoder is independent from TransformerEncoderLayer by passing in the relevant parameters of the layer. The specific differences are described below:

| mindspore.nn.transformer.TransformerEncoder | torch.nn.TransformerEncoder | Descriptions       |
| ---------------------------------------- | --------------------------- | --------------------------------------------------------- |
| batch_size                               |                             | MindSpore requires an additional batch size to be passed in for checksum and incremental inference. |
| num_layers                               | num_layers                  | Same meaning.                                                |
| hidden_size                              | d_model                     | The parameter names are inconsistent and have the same meaning.                                |
| ffn_hidden_size                          |                             |                                                           |
| seq_length                               |                             | encoder input sequence length.                                     |
| num_heads                                |                             |                                                           |
| hidden_dropout_rate                      |                             | hidden_dropout_rate indicates the dropout at the hidden layer.              |
| attention_dropout_rate                   |                             | The attention_dropout_rate indicates the dropout at the softmax.          |
| post_layernorm_residual                  |                             | This parameter of MindSpore indicates whether layernorm is applied to the input when the residuals are summed.  |
| hidden_act                               | activation                  | The type of the activation layer with the same meaning. MindSpore only supports strings.           |
| layernorm_compute_type                   |                             | Controls the calculation type of layernorm.                                 |
| softmax_compute_type                     |                             | Control the calculation type of softmax in attention.                        |
| param_init_type                          |                             | Controls the type of parameter initialization.                                    |
| use_past                                 |                             | Whether to use incremental inference.                                        |
| lambda_func                              |                             | See the API documentation for details on the configuration related to controlling parallelism.                         |
| offset                                   |                             | encoder is used to calculate the initial value of the fusion token.                       |
| moe_config                               |                             | Configuration parameters for MoE parallelism.                                       |
| parallel_config                          |                             | Configuration parameters for parallel settings.                                      |
|                                          | norm                        | Whether to apply the passed norm cell in the output of encoder.                  |

- mindspore.nn.transformer.TransformerEncoder is missing input for src_key_padding_mask.
- mindspore.nn.transformer.TransformerEncoder provides incremental inference functions for static graphs.
- mindspore.nn.transformer.TransformerEncoder uses fp16 for matrix operations by default.
- encoder_mask in mindspore.nn.transformer.TransformerEncoder input is mandatory.
- mindspore.nn.transformer.TransformerEncoder returns the history values of key and value for each layer of attention in encoder.
- The initialization parameters of mindspore.nn.transformer.TransformerEncoder are missing the norm input parameter in torch.nn.Transformer.
- mindspore.nn.transformer.TransformerEncoder provides parallel configuration parallel_config input, which can implement mixed parallelism and pipeline parallelism.

PyTorch: Instantiating the TransformerEncoder requires fewer parameters to be provided.

MindSpore: During class initialization, additional information such as batch_size, source sequence length, etc. needs to be provided, and encoder_mask needs to be input during computation.

## Code Example

```python
import numpy as np
import mindspore as ms
from mindspore.nn.transformer import TransformerEncoder
model = TransformerEncoder(batch_size=32, num_layers=2, hidden_size=512,
                           ffn_hidden_size=2048, seq_length=10, num_heads=8)
encoder_input_value = ms.Tensor(np.random.rand(32, 10, 512), ms.float32)
encoder_input_mask = ms.Tensor(np.ones((32, 10, 10)), ms.float16)
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
