# 比较与torch.nn.ParameterList的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ParameterTuple.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.nn.ParameterList

```python
class torch.nn.ParameterList(parameters=None)
```

更多内容详见[torch.nn.ParameterList](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.ParameterList)。

## mindspore.ParameterTuple

```python
class mindspore.ParameterTuple()
```

更多内容详见[mindspore.ParameterTuple](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore/mindspore.ParameterTuple.html#mindspore.ParameterTuple)。

## 使用方式

PyTorch：以列表形式储存网络参数。

MindSpore：以元组形式储存网络参数。