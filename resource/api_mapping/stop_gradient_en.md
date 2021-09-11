# Comparing the functional differences with torch.autograd.enable_grad and torch.autograd.no_grad

## torch.autograd.enable_grad

```python
torch.autograd.enable_grad()
```

## torch.autograd.no_grad

```python
torch.autograd.no_grad()
```

## mindspore.ops.stop_gradient

```python
mindspore.ops.stop_gradient(input)
```

## Differences

PyTorch: Use `torch.autograd.enable_grad` to enable gradient calculation, and `torch.autograd.no_grad` to disable gradient calculation.

MindSpore: Use [`stop_gradient`](https://www.mindspore.cn/tutorials/en/r1.5/autograd.html#stop-gradient) to disable calculation of gradient for certain operators.
