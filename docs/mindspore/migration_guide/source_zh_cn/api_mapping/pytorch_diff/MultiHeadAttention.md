# 比较与torch.nn.MultiheadAttention的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/MultiHeadAttention.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.MultiheadAttention

```python
import torch
query = torch.rand(10, 32, 512)
key = torch.rand(10, 32, 512)
value = torch.rand(10, 32, 512)
multihead_attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8)
attn_output, attn_output_weights = multihead_attn(query, key, value)
```

更多内容详见[torch.nn.MultiheadAttention](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.MultiheadAttention)。

## mindspore.parallel.nn.MultiHeadAttention

```python
class mindspore.parallel.nn.MultiHeadAttention(*args, **kwargs)(
    query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
    value_past=None, batch_valid_length=None
)
```

更多内容详见[mindspore.parallel.nn.MultiHeadAttention](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.parallel.nn.html#mindspore.parallel.nn.MultiHeadAttention)。

## 使用方式

mindspore.parallel.nn.MultiHeadAttention在初始化参数和torch.nn.MultiheadAttention并不完全相同，但是基本功能保持一致。具体的区别如下说明：

- mindspore.parallel.nn.MultiHeadAttention的dropout_rate分别为0.1，而pytorch对应的版本为0.0。
- mindspore.parallel.nn.MultiHeadAttention缺少bias、add_bias_kv、add_zero_attn、kdim和vdim的输入。
- mindspore.parallel.nn.MultiHeadAttention提供了静态图的增量推理功能。
- mindspore.parallel.nn.MultiHeadAttention默认采用fp16进行矩阵运算。
- mindspore.parallel.nn.MultiHeadAttention的输入中attention_mask是必须的。
- mindspore.parallel.nn.MultiHeadAttention会返回attention的key和value的历史值，而torch.nn.MultiheadAttention可以控制是否返回attention中计算的key和query之间的得分。
- mindspore.parallel.nn.MultiHeadAttention提供了并行配置parallel_config入参，可以实现混合并行。
- mindspore.parallel.nn.MultiHeadAttention的返回tensor的第0维是batch维度，而pytorch则是seq_length为第0维度。

PyTorch：实例化Transformer时需要提供的参数较少。

MindSpore：在类初始化的时候，需要提供batch_size、源序列和目标序列长度等额外信息，并且在计算时需要输入attention_mask。

## 代码示例

```python
import numpy as np
from mindspore.parallel.nn import MultiHeadAttention
from mindspore import dtype as mstype
from mindspore import Tensor
model = MultiHeadAttention(batch_size=32, hidden_size=512, src_seq_length=10, tgt_seq_length=20,
                           num_heads=8)
query = Tensor(np.random.rand(32, 10, 512), mstype.float32)
key = Tensor(np.random.rand(32, 20, 512), mstype.float32)
value = Tensor(np.random.rand(32, 20, 512), mstype.float32)
attention_mask = Tensor(np.ones((32, 10, 20)), mstype.float16)
attn_out, past = model(query, key, value, attention_mask)
# output:
# (32, 20, 512)

import torch
query = torch.rand(20, 32, 512)
key = torch.rand(10, 32, 512)
value = torch.rand(10, 32, 512)
multihead_attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8)
attn_output, attn_output_weights = multihead_attn(query, key, value)
# Out：
# torch.Size([20, 32, 512])
```