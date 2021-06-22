# 前端语法

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/frontend_syntax.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q：导出MindIR格式的时候，`input=np.random.uniform(...)`是不是固定格式？**</font>

A：不是固定格式的，这一步操作是为了创建一个输入，以便于构建网络结构。`export`里只要传入正确的`shape`即可，使用`np.ones`和`np.zeros`创建都是可以的。

<br/>

<font size=3>**Q：MindSpore如何进行参数（如dropout值）修改？**</font>

A：在构造网络的时候可以通过 `if self.training: x = dropput(x)`，验证的时候，执行前设置`network.set_train(mode_false)`，就可以不适用dropout，训练时设置为True就可以使用dropout。

<br/>

<font size=3>**Q：如何查看模型参数量？**</font>

A：可以直接加载CheckPoint统计，可能额外统计了动量和optimizer中的变量，需要过滤下相关变量。
您可以参考如下接口统计网络参数量:

```python
def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params
```

具体[脚本链接](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/cv/tinynet/src/utils.py)。

<br/>

<font size=3>**Q：如何在训练过程中监控`loss`在最低的时候并保存训练参数？**</font>

A：可以自定义一个`Callback`。参考`ModelCheckpoint`的写法，此外再增加判断`loss`的逻辑：

```python
class EarlyStop(Callback):
def __init__(self):
    self.loss = None
def step_end(self, run_context):
     loss =  ****(get current loss)
     if (self.loss == None or loss < self.loss):
         self.loss = loss
         # do save ckpt
```

<br/>

<font size=3>**Q：使用`nn.Conv2d`时，怎样获取期望大小的`feature map`？**</font>

A：`Conv2d shape`推导方法可以[参考这里](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d)，`Conv2d`的`pad_mode`改成`same`，或者可以根据`Conv2d shape`推导公式自行计算`pad`，想要使得`shape`不变，一般pad为`(kernel_size-1)//2`。

<br/>

<font size=3>**Q：使用MindSpore可以自定义一个可以返回多个值的loss函数？**</font>

A：自定义`loss function`后还需自定义`TrainOneStepCell`，实现梯度计算时`sens`的个数和`network`的输出个数相同。具体可参考：

```python
net = Net()

loss_fn = MyLoss()

loss_with_net = MyWithLossCell(net, loss_fn)

train_net = MyTrainOneStepCell(loss_with_net, optim)

model = Model(net=train_net, loss_fn=None, optimizer=None)
```

<br/>

<font size=3>**Q：MindSpore如何实现早停功能？**</font>

A：可以自定义`callback`方法实现早停功能。
例子：当loss降到一定数值后，停止训练。

```python
class EarlyStop(Callback):
    def __init__(self, control_loss=1):
        super(EarlyStep, self).__init__()
        self._control_loss = control_loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if loss.asnumpy() < self._control_loss:
            # Stop training
            run_context._stop_requested = True

stop_cb = EarlyStop(control_loss=1)
model.train(epoch_size, ds_train, callbacks=[stop_cb])
```

<br/>

<font size=3>**Q：模型已经训练好，如何将模型的输出结果保存为文本或者`npy`的格式？**</font>

A：您好，我们网络的输出为`Tensor`，需要使用`asnumpy()`方法将`Tensor`转换为`numpy`，再进行下一步保存。具体可参考：

```python
out = net(x)

np.save("output.npy", out.asnumpy())
```

<br/>

<font size=3>**Q：我用MindSpore在GPU上训练的网络脚本可以不做修改直接在NPU上进行训练么？**</font>

A：可以的，MindSpore面向NPU/GPU/CPU提供统一的API，在算子支持的前提下，网络脚本可以不做修改直接跨平台运行。
