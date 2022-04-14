# Function Differences with torch.optim.Optimizer.step

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TrainOneStepCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torch.optim.Optimizer.step

```python
torch.optim.Optimizer.step(closure)
```

For more information, see [torch.optim.Optimizer.step](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.Optimizer.step).

## mindspore.nn.TrainOneStepCell

```python
class mindspore.nn.TrainOneStepCell(
    network,
    optimizer,
    sens=1.0
)((*inputs))
```

For more information, see [mindspore.nn.TrainOneStepCell](https://mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.TrainOneStepCell.html#mindspore.nn.TrainOneStepCell).

## Differences

PyTorch: An abstract method of the abstract class `Optimizer`, and it should be inherited and implemented by `Optimizer`'s subclass and return loss.

MindSpore: A class, which requires `network` and `optimizer` to be passed as parameters, and loss will be returned by the `construct` method.