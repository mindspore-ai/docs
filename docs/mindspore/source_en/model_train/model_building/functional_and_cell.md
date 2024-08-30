# Functional and Cell

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/model_building/functional_and_cell.md)

## Operator Functional Interface

## nn Subnet Interface

### Cell Training State Change

Some Tensor operations in neural networks do not behave the same during training and inference, e.g., `nn.Dropout` performs random dropout during training but not during inference, and `nn.BatchNorm` requires updating the `mean` and `var` variables during training and fixing their values unchanged during inference. So we can set the state of the neural network through the `Cell.set_train` interface.

When `set_train` is set to True, the neural network state is `train`, and the default value of `set_train` interface is `True`:

```python
   net.set_train()
   print(net.phase)
```

```text
   train
```

When `set_train` is set to False, the neural network state is `predict`:

```python
   net.set_train()
   print(net.phase)
```

```text
   predict
```