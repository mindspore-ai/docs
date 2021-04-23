# Function Differences with torch.cuda.set_device

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