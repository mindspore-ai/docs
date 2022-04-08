# 比较与torch.nn.MultiheadAttention的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MultiHeadAttention.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.MultiheadAttention

```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
```

更多内容详见[torch.nn.MultiheadAttention](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MultiheadAttention)。

## mindspore.nn.transformer.MultiHeadAttention

```python
class mindspore.nn.transformer.MultiHeadAttention(batch_size, src_seq_length, tgt_seq_length, hidden_size, num_heads, hidden_dropout_rate=0.1, attention_dropout_rate=0.1, compute_dtype=mstype.float16, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, use_past=False, parallel_config=default_dpmp_config)(
    query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
    value_past=None, batch_valid_length=None
)
```

更多内容详见[mindspore.nn.transformer.MultiHeadAttention](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.transformer.html#mindspore.nn.transformer.MultiHeadAttention)。

## 使用方式

mindspore.nn.transformer.MultiHeadAttention在初始化参数和torch.nn.MultiheadAttention并不完全相同，但是基本功能保持一致。具体的区别如下说明：

| mindspore.nn.transformer.MultiHeadAttention | torch.nn.MultiheadAttention | 说明                                                         |
| ---------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| batch_size                               |                             | MindSpore需要传入额外的batch size以作校验和增量推理使用。    |
| src_seq_length                           |                             | encoder输入序列长度。                                        |
| tgt_seq_length                           |                             | decoder输入序列长度。                                        |
| hidden_size                              | embed_dim                   | 参数名称不一致，含义相同。                                   |
| num_heads                                | num_heads                   | Attention的head数目，含义相同。                              |
| hidden_dropout_rate                      | dropout                     | 含义不同。hidden_dropout_rate表示在隐藏层处的dropout，而PyTorch的dropout参数额外控制了softmax处的dropout rate。 |
| attention_dropout_rate                   | dropout                     | 含义不同。attention_dropout_rate表示在softmax处的dropout，而PyTorch的dropout参数额外控制了隐藏层的dropout rate。 |
| compute_type                             |                             | 控制内部matmul矩阵计算类型。                                 |
| softmax_compute_type                     |                             | 控制attention中softmax的计算类型。                           |
| param_init_type                          |                             | 控制参数初始化的类型。                                       |
| use_past                                 |                             | 是否使用增量推理。                                           |
| parallel_config                          |                             | 并行设置的配置参数。                                         |
|                                          | bias                        | 是否在projection层添加bias。MindSpore默认行为是添加的。      |
|                                          | add_bias_kv                 | 是否在第0维度的key和value序列上面添加bias。MindSpore未实现此功能。 |
|                                          | add_zero_attn               | 是否在第1维度的key和value序列上面添加全零的数据。MindSpore未实现此功能。 |
|                                          | kdims                       | key维度的feature数量。MindSpore未实现此功能。                |
|                                          | vdims                       | value维度的feature数量。MindSpore未实现此功能。              |
|                                          | batch_first                 | MindSpore默认配置为(batch,seq, feature)，即Pytorch的batch_first=True |

- mindspore.nn.transformer.MultiHeadAttention缺少bias、add_bias_kv、add_zero_attn、kdim和vdim的输入，在前向计算中缺少key_padding_mask的输入。
- mindspore.nn.transformer.MultiHeadAttention的dropout_rate分别为0.1，而pytorch对应的版本为0.0。
- mindspore.nn.transformer.MultiHeadAttention的输入中attention_mask是必须的。
- mindspore.nn.transformer.MultiHeadAttention提供了并行配置parallel_config入参，可以实现混合并行。
- mindspore.nn.transformer.MultiHeadAttention会返回attention的key和value的历史值，而torch.nn.MultiheadAttention可以控制是否返回attention中计算的key和query之间的得分。
- mindspore.nn.transformer.MultiHeadAttention提供了静态图的增量推理功能。
- mindspore.nn.transformer.MultiHeadAttention默认采用fp16进行矩阵运算。
- mindspore.nn.transformer.MultiHeadAttention的返回tensor的第0维是batch维度，而pytorch默认是第0维度为seq_length。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列和目标序列长度等额外信息，并且在计算时需要输入attention_mask。

## 代码示例

```python
import numpy as np
from mindspore.nn.transformer import MultiHeadAttention
from mindspore import dtype as mstype
from mindspore import Tensor
model = MultiHeadAttention(batch_size=32, hidden_size=512, src_seq_length=10, tgt_seq_length=20,
                           num_heads=8)
query = Tensor(np.random.rand(32, 10, 512), mstype.float32)
key = Tensor(np.random.rand(32, 20, 512), mstype.float32)
value = Tensor(np.random.rand(32, 20, 512), mstype.float32)
attention_mask = Tensor(np.ones((32, 10, 20)), mstype.float16)
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