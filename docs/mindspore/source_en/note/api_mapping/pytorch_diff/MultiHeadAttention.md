# Function Differences with torch.nn.MultiheadAttention

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MultiHeadAttention.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.MultiheadAttention

```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
```

For more information, see [torch.nn.MultiheadAttention](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MultiheadAttention).

## mindspore.nn.transformer.MultiHeadAttention

```python
class mindspore.nn.transformer.MultiHeadAttention(batch_size, src_seq_length, tgt_seq_length, hidden_size, num_heads, hidden_dropout_rate=0.1, attention_dropout_rate=0.1, compute_dtype=mstype.float16, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, use_past=False, parallel_config=default_dpmp_config)(
    query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
    value_past=None, batch_valid_length=None
)
```

For more information, see [mindspore.nn.transformer.MultiHeadAttention](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.MultiHeadAttention).

## Usage

torch.nn.MultiHeadAttention is not exactly the same as torch.nn.transformer.MultiheadAttention in terms of initialization parameters, but the basic function remains the same. The specific differences are described below:

| mindspore.nn.transformer.MultiHeadAttention | torch.nn.MultiheadAttention | Descriptions         |
| ---------------------------------------- | --------------------------- | ------|
| batch_size                    |                             | MindSpore requires an additional batch size to be passed in for checksum and incremental inference.    |
| src_seq_length                           |                             | encoder input sequence length.                                        |
| tgt_seq_length                           |                             | decoder input sequence length.                                        |
| hidden_size                              | embed_dim                   | Inconsistent parameter names, the same meaning.                                   |
| num_heads                                | num_heads                   | Number of heads of Attention, same meaning.                              |
| hidden_dropout_rate                      | dropout                     | The meaning is different. hidden_dropout_rate indicates the dropout at the hidden layer, while PyTorch dropout parameter additionally controls the dropout rate at the softmax. |
| attention_dropout_rate                   | dropout                     | The meaning is different. attention_dropout_rate indicates the dropout at softmax, while PyTorch dropout parameter additionally controls the dropout rate of the hidden layer. |
| compute_type                             |                             | Controls the type of internal matmul matrix calculation.          |
| softmax_compute_type                |            | Control the type of softmax calculation in attention.                |
| param_init_type                          |                             | Control the type of parameter initialization.                                       |
| use_past                                 |                             | Whether to use incremental inference.                                           |
| parallel_config                          |                             | Configuration parameters for parallel settings.                                         |
|                                          | bias                        | Whether to add bias to the projection layer. The default behavior of MindSpore is to add it.      |
|                                          | add_bias_kv                 | Whether to add bias on top of the key and value sequence in the 0th dimension. MindSpore does not implement this feature. |
|                                          | add_zero_attn               | Whether to add bias on top of the key and value sequence in the first dimension. MindSpore does not implement this feature. |
|                                          | kdims                       | The number of features of the key dimension. MindSpore does not implement this feature.                |
|                                          | vdims                       | The number of features of the key dimension. MindSpore does not implement this feature.             |
|                                          | batch_first                 | MindSpore is configured as (batch,seq, feature) by default, i.e. PyTorch's batch_first=True |

- mindspore.nn.transformer.MultiHeadAttention is missing the input of bias, add_bias_kv, add_zero_attn, kdim and vdim, and missing the input of key_padding_mask in the forward calculation.
- The dropout_rate of mindspore.nn.transformer.MultiHeadAttention is 0.1 respectively, while the corresponding version of pytorch is 0.0.
- attention_mask is required in the input of mindspore.nn.transformer.MultiHeadAttention.
- mindspore.nn.transformer.MultiHeadAttention provides parallel configuration parallel_config inputs to enable mixed parallelism.
- mindspore.nn.transformer.MultiHeadAttention returns the history values of key and value in attention, while torch.nn.MultiheadAttention controls whether to return the key and query computed in attention.
- mindspore.nn.transformer.MultiHeadAttention provides incremental inference for static graphs.
- mindspore.nn.transformer.MultiHeadAttention uses fp16 for matrix operations by default.
- mindspore.nn.transformer.MultiHeadAttention returns the zero dimension of the tensor as the batch dimension, while PyTorch defaults to the zero dimension of seq_length.

PyTorch: Instantiating the Transformer requires fewer parameters to be provided.

MindSpore: During class initialization, additional information such as batch_size, source and target sequence lengths need to be provided, and attention_mask needs to be input during computation.

## Code Example

```python
import numpy as np
from mindspore.nn.transformer import MultiHeadAttention
import mindspore as ms
model = MultiHeadAttention(batch_size=32, hidden_size=512, src_seq_length=10, tgt_seq_length=20,
                           num_heads=8)
query = ms.Tensor(np.random.rand(32, 10, 512), ms.float32)
key = ms.Tensor(np.random.rand(32, 20, 512), ms.float32)
value = ms.Tensor(np.random.rand(32, 20, 512), ms.float32)
attention_mask = ms.Tensor(np.ones((32, 10, 20)), ms.float16)
output, past = model(query, key, value, attention_mask)
print(output.shape)
# output:
# (32, 20, 512)

import torch
query = torch.rand(20, 32, 512)
key = torch.rand(10, 32, 512)
value = torch.rand(10, 32, 512)
multihead_attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8)
output, attn_output_weights = multihead_attn(query, key, value)
print(output.shape)
# output:
# torch.Size([20, 32, 512])
```
