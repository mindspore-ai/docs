# Comparing the functional differences with torch.autograd.enable_grad and torch.autograd.no_grad

## torch.autograd.enable_grad

```python
torch.autograd.enable_grad()
```

For more information, see[torch.autograd.enable_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.enable_grad).

## torch.autograd.no_grad

```python
torch.autograd.no_grad()
```

For more information, see[torch.autograd.no_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.no_grad).

## mindspore.ops.stop_gradient

```python
mindspore.ops.stop_gradient(input)
```

For more information, see[mindspore.ops.stop_gradient](https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.ops.html#functional).

## Differences

PyTorch: Use `torch.autograd.enable_grad` to enable gradient calculation, and `torch.autograd.no_grad` to disable gradient calculation.

MindSpore: Use [stop_gradient](https://www.mindspore.cn/tutorials/en/r1.3/autograd.html#stop-gradient) to disable calculation of gradient for certain operators.
