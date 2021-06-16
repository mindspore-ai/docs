# Function Differences with torch.optim.Optimizer.step

## torch.optim.Optimizer.step

```python
torch.optim.Optimizer.step()
```

## mindspore.nn.TrainOneStepCell

```python
class mindspore.nn.TrainOneStepCell(
    network,
    optimizer,
    sens=1.0
)((*inputs))
```

## Differences

PyTorch: An abstract method of the abstract class `Optimizer`, and it should be inherited and implemented by `Optimizer`'s subclass and return loss.

MindSpore: A class, which requires `network` and `optimizer` to be passed as parameters, and loss will be returned by the `construct` method.