# Function Differences with torch.nn.TransformerDecoder

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TransformerDecoder.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.TransformerDecoder

```python
torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
```

For more information, see [torch.nn.TransformerDecoder](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.TransformerDecoder).

## mindspore.nn.transformer.TransformerDecoder

```python
class mindspore.nn.transformer.TransformerDecoder(num_layers, batch_size, hidden_size, ffn_hidden_size, src_seq_length, tgt_seq_length, num_heads, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", lambda_func=None, use_past=False, offset=0, moe_config=default_moe_config, parallel_config=default_transformer_config)(
    hidden_stats, decoder_mask, encoder_output=None,
    memory_mask=None, init_reset=True, batch_valid_length=None
)
```

For more information, see [mindspore.nn.transformer.TransformerDecoder](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.TransformerDecoder).

## Usage

The initialization parameters of mindspore.nn.transformer.TransformerDecoder and torch.nn.TransformerDecoder are not exactly the same, but the basic functions remain the same. The torch.nn.TransformerDecoder uses the combination method, i.e., the instantiated TransformerDecoderLayer is used as the input parameter of torch.nn.TransformerDecoder. TransformerDecoder is independent of TransformerDecoderLayer by passing in the parameters of the layer. The specific differences are explained as follows:

| mindspore.nn.transformer.TransformerDecoder | torch.nn.TransformerDecoder | Descriptions                                                      |
| ---------------------------------------- | --------------------------- | --------------------------------------------------------- |
| num_layers                               | num_layers                  | Same meaning.                                                |
| batch_size                               |                             | MindSpore requires an additional batch size to be passed in for checksum and incremental inference. |
| hidden_size                              |                             | The parameter names are inconsistent and have the same meaning.               |
| ffn_hidden_size  |                 |         |
| src_seq_length                           |                             | encoder input sequence length.                                     |
| tgt_seq_length                           |                             | decoder input sequence length.                                     |
| num_heads                                |                             |                                                           |
| attention_dropout_rate                   |                             | The attention_dropout_rate indicates the dropout at the softmax.          |
| hidden_dropout_rate                      |                             | hidden_dropout_rate indicates the dropout at the hidden layer.              |
| post_layernorm_residual                  |                             | This parameter of MindSpore indicates whether layernorm is applied to the input when the residuals are summed.  |
| layernorm_compute_type                   |                             | Control the calculation type of layernorm.                                 |
| softmax_compute_type                     |                             | Control the calculation type of softmax in attention.                        |
| param_init_type                          |                             | Control the type of parameter initialization.           |
| hidden_act                               |                             | The type of the activation layer with the same meaning. MindSpore only supports strings.           |
| lambda_func                              |                             | See the API documentation for details on the configuration related to controlling parallelism.                         |
| use_past          |       | Whether to use incremental inference.       |
| offset     |        | encoder is used to calculate the initial value of the fusion token.  |
| moe_config         |         | Configuration parameters for MoE parallelism.    |
| parallel_config       |     | Configuration parameters for parallel settings.   |
|               | decoder_layer  | Instantiation parameters for decoder     |
|         | norm   | Whether to apply the incoming norm cell in the output of the decoder.|

- mindspore.nn.transformer.TransformerDecoder is missing input of tgt_key_padding_mask and emory_key_padding_mask.
- mindspore.nn.transformer.TransformerDecoder provides incremental inference functions for static graphs.
- mindspore.nn.transformer.TransformerDecoder uses fp16 for matrix operation by default.
- The input of mindspore.nn.transformer.TransformerDecoder attention_mask is required to be input.
- mindspore.nn.transformer.TransformerDecoder returns the history values of key, value for each level of attention in encoder and decoder.
- mindspore.nn.transformer.TransformerDecoder is missing the norm input parameter in torch.nn.Transformer in its initialization parameters.
- mindspore.nn.transformer.TransformerDecoder provides parallel configuration parallel_config input parameter, which can realize mixed parallelism and pipeline parallelism.

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
