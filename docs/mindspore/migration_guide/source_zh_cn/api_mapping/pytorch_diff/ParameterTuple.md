# 比较与torch.nn.ParameterList的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ParameterTuple.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.ParameterList

```python
class torch.nn.ParameterList()
```

## mindspore.ParameterTuple

```python
class mindspore.ParameterTuple()
```

## 使用方式

PyTorch: 以列表形式储存网络参数。

MindSpore：以元组形式储存网络参数。