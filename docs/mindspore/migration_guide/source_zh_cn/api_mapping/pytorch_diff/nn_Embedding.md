# 比较与torch.nn.Embedding的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/nn_Embedding.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.Embedding

```python
class torch.nn.Embedding(
    num_embeddings,
    embedding_dim,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    _weight=None)
```

更多内容详见[torch.nn.Embedding](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Embedding)。

## mindspore.nn.Embedding

```python
class mindspore.nn.Embedding(
    vocab_size,
    embedding_size,
    use_one_hot=False,
    embedding_table="normal",
    dtype=mstype.float32,
    padding_idx=None)
```

更多内容详见[mindspore.nn.Embedding](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Embedding.html#mindspore.nn.Embedding)。

## 使用方式

PyTorch：支持使用`_weight`属性初始化embedding，并且可以通过`weight`变量获取当前embedding的权重。

MindSpore：支持使用`embedding_table`属性初始化embedding，并且可以通过`embedding_table`属性获取当前embedding的权重。除此之外，当`use_one_hot`为True时，你可以得到具有one-hot特征的embedding。

## 代码示例

```python
import mindspore
import torch
import numpy as np

# In MindSpore, you can initialize the embedding weight with embedding_table attribute.
embedding_weight = mindspore.Tensor(np.random.randn(3, 5), dtype=mindspore.float32)
print(embedding_weight)
# Out:
# [[-1.6178044   0.7513232   2.3800876  -0.9759835   0.21116002]
#  [-0.10350747  1.6457493   0.456487    1.6116067   0.870742  ]
#  [ 0.02000403  0.05202193 -0.15820664  0.6536981   0.6578212 ]]
embedding = mindspore.nn.Embedding(vocab_size=embedding_weight.shape[0],
                                   embedding_size=embedding_weight.shape[1],
                                   embedding_table=embedding_weight)
x = mindspore.Tensor([0, 1], dtype=mindspore.int32)
result = embedding(x)
print(result)
# Out:
# [[-1.6178044   0.7513232   2.3800876  -0.9759835   0.21116002]
#  [-0.10350747  1.6457493   0.456487    1.6116067   0.870742  ]]

# In MindSpore, you can get the embedding weight with the embedding_table attribute.
print(embedding.embedding_table)
# Out:
# Parameter (name=embedding_table, shape=(3, 5), dtype=Float32, requires_grad=True)

# In MindSpore, you can get an embedding with the feature of one-hot embedding when use_one_hot is True.
embedding = mindspore.nn.Embedding(vocab_size=3,
                                   embedding_size=3,
                                   use_one_hot=True)
x = mindspore.Tensor([0, 1, 2], dtype=mindspore.int32)
result = embedding(x)
print(result)
# Out:
# [[ 0.01174604 -0.00526122 -0.00539862]
#  [-0.00962828 -0.01533093 -0.01377784]
#  [-0.01083433  0.00337794 -0.00224762]]

# In Pytorch, you can initialize the embedding weight with _weight attribute.
embedding_weight = torch.randn([3, 5], dtype=torch.float32)
print(embedding_weight)
# Out：
# tensor([[ 0.2546, -0.9063, -0.3263,  0.4768, -0.6208],
#         [-0.8473, -1.2814, -1.6156, -0.8399, -0.0408],
#         [ 1.5786,  0.0389, -0.5644,  1.8605,  0.6947]])
embedding = torch.nn.Embedding(num_embeddings=embedding_weight.size()[0],
                               embedding_dim=embedding_weight.size()[1],
                               _weight=embedding_weight)
x = torch.tensor([0, 1])
result = embedding(x)
print(result)
# Out：
# tensor([[ 0.2546, -0.9063, -0.3263,  0.4768, -0.6208],
#         [-0.8473, -1.2814, -1.6156, -0.8399, -0.0408]],
#        grad_fn=<EmbeddingBackward>)

# In Pytorch, you can get the embedding weight with the weight variable.
print(embedding.weight)
# Out：
# Parameter containing:
# tensor([[ 0.2546, -0.9063, -0.3263,  0.4768, -0.6208],
#         [-0.8473, -1.2814, -1.6156, -0.8399, -0.0408],
#         [ 1.5786,  0.0389, -0.5644,  1.8605,  0.6947]], requires_grad=True)
```
