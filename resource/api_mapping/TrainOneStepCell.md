# 比较与torch.optim.Optimizer.step的功能差异

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

## 使用方式

PyTorch: 是`Optimizer`这个抽象类的抽象方法，需要由`Optimizer`的子类继承后具体实现，返回损失值。

MindSpore：是1个类，需要把`network`和`optimizer`作为参数传入，且需要调用`construct`方法返回损失值。