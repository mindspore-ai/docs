# 后端运行类

`Ascend` `GPU` `CPU` `环境准备` `运行模式` `模型训练` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/backend_running.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q：MindSpore如何实现早停功能？

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

Q：请问自己制作的黑底白字`28*28`的数字图片，使用MindSpore训练出来的模型做预测，报错提示`wrong shape of image`是怎么回事？

A：首先MindSpore训练使用的灰度图MNIST数据集。所以模型使用时对数据是有要求的，需要设置为`28*28`的灰度图，就是单通道才可以。

<br/>

Q：MindSpore的operation算子报错：`device target [CPU] is not supported in pynative mode`

A：pynative 模式目前只支持Ascend和GPU，暂时还不支持CPU。

<br/>

Q：在Ascend平台上，执行用例有时候会报错run task error，如何获取更详细的日志帮助问题定位？

A：可以通过开启slog获取更详细的日志信息以便于问题定位，修改`/var/log/npu/conf/slog/slog.conf`中的配置，可以控制不同的日志级别，对应关系为：0:debug、1:info、2:warning、3:error、4:null(no output log)，默认值为1。

<br/>

Q：使用ExpandDims算子报错：`Pynative run op ExpandDims failed`。具体代码：

```python
context.set_context(
mode=cintext.GRAPH_MODE,
device_target='ascend')
input_tensor=Tensor(np.array([[2,2],[2,2]]),mindspore.float32)
expand_dims=ops.ExpandDims()
output=expand_dims(input_tensor,0)
```

A：这边的问题是选择了Graph模式却使用了PyNative的写法，所以导致报错，MindSpore支持两种运行模式，在调试或者运行方面做了不同的优化:

- PyNative模式：也称动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。

- Graph模式：也称静态图模式或者图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。

用户可以参考[官网教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/debug_in_pynative_mode.html)选择合适、统一的模式和写法来完成训练。

<br/>

Q：使用Ascend平台执行训练过程，出现报错：`Out of Memory!!! total[3212254720] (dynamic[0] memory poll[524288000]) malloc[32611480064] failed!` 如何解决？

A：此问题属于内存占用过多导致的内存不够问题，可能原因有两种：

- `batch_size`的值设置过大。解决办法：将`batch_size`的值设置减小。
- 引入了异常大的`Parameter`，例如单个数据shape为[640,1024,80,81]，数据类型为float32，单个数据大小超过15G，这样差不多大小的两个数据相加时，占用内存超过3*15G，容易造成`Out of Memory`。解决办法：检查参数的`shape`，如果异常过大，减少shape。
- 如果以上操作还是未能解决，可以上[官方论坛](https://bbs.huaweicloud.com/forum/forum-1076-1.html)发帖提出问题，将会有专门的技术人员帮助解决。
