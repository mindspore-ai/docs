# Function Differences with torch.nn.Transformer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Transformer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Transformer

```python
torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
```

For more information, see [torch.nn.Transformer](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Transformer).

## mindspore.nn.transformer.Transformer

```python
class mindspore.nn.transformer.Transformer(hidden_size, batch_size, ffn_hidden_size, src_seq_length, tgt_seq_length, encoder_layers=3, decoder_layers=3, num_heads=2, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu", post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, lambda_func=None, use_past=False, moe_config=default_moe_config, parallel_config=default_transformer_config)(
    encoder_inputs, encoder_masks, decoder_inputs=None,
    decoder_masks=None, memory_mask=None, init_reset=True, batch_valid_length=None
)
```

For more information, see [mindspore.nn.transformer.Transformer](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.Transformer).

## Usage

mindspore.nn.transformer.Transformer is not exactly the same as torch.nn.Transformer in terms of initialization parameters, but the basic function remains the same. It is because mindspore.nn.Transformer provides more fine-grained control as well as parallel configuration, which can easily enable parallel training. The main differences are summarized as follows:

| mindspore.nn.transformer.Transformer | torch.nn.Transformer | Descriptions  |
| --------------------- | --------- | ------------------------- |
| hidden_size      | d_model              | The parameter names are inconsistent and have the same meaning.        |
| batch_size       |                      | MindSpore requires an additional batch size to be passed in for checksum and incremental inference.    |
| ffn_hidden_size                   | dim_feedforward      | The parameter names are inconsistent and have the same meaning.      |
| src_seq_length                    |                      | encoder input sequence length.     |
| tgt_seq_length                    |                      | decoder input sequence length.          |
| encoder_layers                    | num_encoder_layers   | The number of layers of encoder, with the same meaning.     |
| decoder_layers                    | num_decoder_layers   | The number of layers of decoder, with the same meaning.       |
| num_heads                         | nhead                | The number of heads of Attention, same meaning.      |
| attention_dropout_rate            | dropout              | The meaning is different. attention_dropout_rate indicates the dropout at softmax, while PyTorch dropout parameter additionally controls the dropout rate of the hidden layer. |
| hidden_dropout_rate               | dropout              | The meaning is different. hidden_dropout_rate indicates the dropout at the hidden layer, while PyTorch dropout parameter additionally controls the dropout rate at the softmax. |
| hidden_act                        | activation           | The type of the activation layer with the same meaning. MindSpore only supports strings.       |
| post_layernorm_residual           | norm_first           | The meaning is different. MindSpore parameter indicates whether the residual summation applies layernorm to the input, while PyTorch indicates whether to input layernorm first before inputting sublayer. |
| layernorm_compute_type            |                      | Control the calculation type of layernorm.              |
| softmax_compute_type              |                      | Control the calculation type of softmax in attention.       |
| param_init_type                   |                      | Controls the type of parameter initialization.      |
| lambda_func                       |                      | See the API documentation for details on the configuration related to controlling parallelism.              |
| use_past                          |                      | Whether to use incremental inference.          |
| moe_config                        |                      | Configuration parameters for MoE parallelism.             |
| parallel_config                   |                      | Configuration parameters for parallel settings.             |
|                                   | custom_encoder       | User-defined encoder.       |
|                                   | custom_decoder       | User-defined decoder.     |
|                                   | layer_norm_eps       | The value of the initial zero is prevented in the layernorm calculation.        |
|                                   | batch_first          | Whether batch is the 0th dimension in input and output Tensor. MindSpore takes the 0th dimension as the batch dimension, which corresponds to setting bathc_first=True in torch.nn.transformer. |

In addition to the above differences in initialization parameters, there are a number of input and output differences for forward execution as follows:

- mindspore.nn.transformer.Transformer is missing src_key_padding_mask, tgt_key_padding_mask and memory_key_padding_mask inputs.

- encoder_mask and decoder_mask in the input of mindspore.nn.transformer.Transformer are mandatory.

- mindspore.nn.transformer.Transformer additionally returns the historical values of key and value for each layer of attention in encoder and decoder.

- Comparison of the parameters of post_layernorm_residual in mindspore.nn.transformer and norm_first in torch.nn.transformer is as follows:

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

In addition mindspore.nn.transformer.Transformer has the following functional differences:

- mindspore.nn.transformer.Transformer provides incremental inference for static graphs.
- mindspore.nn.transformer.Transformer uses fp16 for matrix operations by default.

PyTorch: Instantiating the Transformer requires fewer parameters to be provided.

MindSpore: During class initialization, additional information such as batch_size, sentence length of source and target sequences, and encoder_mask and decoder_mask need to be input during computation.

## Code Example

```python
import numpy as np
import mindspore as ms
from mindspore.nn.transformer import Transformer

model = Transformer(batch_size=32, encoder_layers=1,
                    decoder_layers=1, hidden_size=512, ffn_hidden_size=2048,
                    src_seq_length=10, tgt_seq_length=20)
encoder_input_value = ms.Tensor(np.random.rand(32, 10, 512), ms.float32)
encoder_input_mask = ms.Tensor(np.ones((32, 10, 10)), ms.float16)
decoder_input_value = ms.Tensor(np.random.rand(32, 20, 512), ms.float32)
decoder_input_mask = ms.Tensor(np.ones((32, 20, 20)), ms.float16)
memory_mask = ms.Tensor(np.ones((32, 20, 10)), ms.float16)
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
