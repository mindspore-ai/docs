# Function Differences with torch.nn.ParameterList

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/ParameterTuple.md)

## torch.nn.ParameterList

```python
class torch.nn.ParameterList(parameters=None)
```

For more information, see [torch.nn.ParameterList](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.ParameterList).

## mindspore.ParameterTuple

```python
class mindspore.ParameterTuple()
```

For more information, see [mindspore.ParameterTuple](https://mindspore.cn/docs/api/en/r1.6/api_python/mindspore/mindspore.ParameterTuple.html#mindspore.ParameterTuple).

## Differences

PyTorch: Stores parameters of network into a list.

MindSpore：Stores parameters of network into a tuple.