# Functional与Cell

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/model_building/functional_and_cell.md)

## 算子Functional接口

## nn子网络接口

### Cell训练状态转换

神经网络中的部分Tensor操作在训练和推理时的表现并不相同，如`nn.Dropout`在训练时进行随机丢弃，但在推理时则不丢弃，`nn.BatchNorm`在训练时需要更新`mean`和`var`两个变量，在推理时则固定其值不变。因此我们可以通过`Cell.set_train`接口来设置神经网络的状态。

`set_train(True)`时，神经网络状态为`train`, `set_train`接口默认值为`True`:

```python
   net.set_train()
   print(net.phase)
```

```text
   train
```

`set_train(False)`时，神经网络状态为`predict`：

```python
   net.set_train()
   print(net.phase)
```

```text
   predict
```