# TrainOneStepCell

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/trainonestepcell.md)

`TrainOneStepCell` is used to perform single-step training of the network and return the loss result after each training result.

The following describes how to build an instance for using the `TrainOneStepCell` API to perform network training. The import code of the `LeNet5` and package name is the same as that in the previous case.

```python
data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.ones([32]).astype(np.int32))
net = LeNet5()
learning_rate = 0.01
momentum = 0.9

optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = WithLossCell(net, criterion)
train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
for i in range(5):
    train_network.set_train()
    res = train_network(data, label)
    print(f"+++++++++result:{i}++++++++++++")
    print(res)
```

```text
+++++++++result:0++++++++++++
2.302585
+++++++++result:1++++++++++++
2.2935712
+++++++++result:2++++++++++++
2.2764661
+++++++++result:3++++++++++++
2.2521412
+++++++++result:4++++++++++++
2.2214084
```

In the case, an optimizer and a `WithLossCell` instance are built, and then a training network is initialized in `TrainOneStepCell`. The case is repeated for five times, that is, the network is trained for five times, and the loss result of each time is output, the result shows that the loss value gradually decreases after each training.

The following content will describe how MindSpore uses more advanced encapsulation APIs, that is, the `train` method in the `Model` class to train a model. Many network components, such as `TrainOneStepCell` and `WithLossCell`, will be used in the internal implementation.
You can view the internal implementation of these components.
