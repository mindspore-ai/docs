# 比较与torch.optim.Optimizer.step的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/TrainOneStepCell.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.optim.Optimizer.step

```python
torch.optim.Optimizer.step(closure)
```

更多内容详见[torch.optim.Optimizer.step](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.Optimizer.step)。

## mindspore.nn.TrainOneStepCell

```python
class mindspore.nn.TrainOneStepCell(
    network,
    optimizer,
    sens=1.0
)((*inputs))
```

更多内容详见[mindspore.nn.TrainOneStepCell](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.TrainOneStepCell.html#mindspore.nn.TrainOneStepCell)。

## 使用方式

PyTorch：是`Optimizer`这个抽象类的抽象方法，需要由`Optimizer`的子类继承后具体实现，返回损失值。

MindSpore：是1个类，需要把`network`和`optimizer`作为参数传入，且需要调用`construct`方法返回损失值。