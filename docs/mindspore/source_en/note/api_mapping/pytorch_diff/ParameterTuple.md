# Function Differences with torch.nn.ParameterList

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ParameterTuple.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.ParameterList

```python
class torch.nn.ParameterList(parameters=None)
```

For more information, see [torch.nn.ParameterList](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.ParameterList).

## mindspore.ParameterTuple

```python
class mindspore.ParameterTuple()
```

For more information, see [mindspore.ParameterTuple](https://mindspore.cn/docs/en/r1.7/api_python/mindspore/mindspore.ParameterTuple.html#mindspore.ParameterTuple).

## Differences

PyTorch: Stores parameters of network into a list.

MindSpore：Stores parameters of network into a tuple.