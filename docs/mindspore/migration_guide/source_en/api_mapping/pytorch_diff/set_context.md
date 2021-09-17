# Function Differences with torch.cuda.set_device

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/set_context.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.cuda.set_device

```python
torch.cuda.set_device(device)
```

## mindspore.context.set_context

```python
mindspore.context.set_context(**kwargs)
```

## Differences

PyTorch: It is used to set the current `device`.

MindSpore：It is not only used to set the current `device`, but also set the `mode`, `device_target`, `save_graphs`, etc.