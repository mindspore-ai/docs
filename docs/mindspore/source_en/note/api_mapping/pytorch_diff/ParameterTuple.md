# Function Differences with torch.nn.ParameterList

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ParameterTuple.md)

## torch.nn.ParameterList

```python
class torch.nn.ParameterList(parameters=None)
```

For more information, see [torch.nn.ParameterList](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.ParameterList).

## mindspore.ParameterTuple

```python
class mindspore.ParameterTuple()
```

For more information, see [mindspore.ParameterTuple](https://mindspore.cn/docs/en/r2.0/api_python/mindspore/mindspore.ParameterTuple.html#mindspore.ParameterTuple).

## Differences

PyTorch: Stores parameters of network into a list.

MindSpore: Stores parameters of network into a tuple.