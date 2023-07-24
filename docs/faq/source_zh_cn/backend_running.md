# 后端运行类

`Ascend` `GPU` `CPU` `环境准备` `运行模式` `模型训练` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_zh_cn/backend_running.md)

<font size=3>**Q：请问`c_transforms`和`py_transforms`有什么区别，比较推荐使用哪个？**</font>

A：推荐使用`c_transforms`，因为纯C层执行，所以性能会更好。

原理:`c_transform`底层使用的是C版本`opencv/jpeg-turbo`进行的数据处理，`py_transform`使用的是Python版本的`Pillow`进行数据处理。

<br/>

<font size=3>**Q：MindSpore在NPU硬件平台进行多卡训练，自定义数据集如何给不同NPU传递不同数据？**</font>

A：使用`GeneratorDataset`的时候，可以使用`num_shards=num_shards`,`shard_id=device_id`参数来控制不同卡读取哪个分片的数据，`__getitem__`和`__len__`按全量数据集处理即可。

举例：

```python
# 卡0：
ds.GeneratorDataset(..., num_shards=8, shard_id=0, ...)
# 卡1：
ds.GeneratorDataset(..., num_shards=8, shard_id=1, ...)
# 卡2：
ds.GeneratorDataset(..., num_shards=8, shard_id=2, ...)
...
# 卡7：
ds.GeneratorDataset(..., num_shards=8, shard_id=7, ...)
```

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

具体[脚本链接](https://gitee.com/mindspore/mindspore/blob/r1.2/model_zoo/research/cv/tinynet/src/utils.py)。

<br/>

<font size=3>**Q：如何构建图像的多标签MindRecord格式数据集？**</font>

A：数据Schema可以按如下方式定义：`cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

说明：label是一个数组，numpy类型，这里面可以存你说的 1， 1，0，1， 0， 1 这么多label值，这些label值对应同一个data，即：同一个图像的二进制值。
可以参考[将数据集转换为MindRecord](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/convert_dataset.html#将数据集转换为MindRecord)教程。

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

<font size=3>**Q：`mindspore/tests`下怎样执行单个`ut`用例？**</font>

A：`ut`用例通常需要基于debug版本的MindSpore包，官网并没有提供。可以基于源码使用`sh build.sh`编译，然后通过`pytest`指令执行，debug模式编包不依赖后端。编译选项`sh build.sh -t on`，用例执行可以参考`tests/runtest.sh`脚本。

<br/>

<font size=3>**Q：使用`nn.Conv2d`时，怎样获取期望大小的`feature map`？**</font>

A：`Conv2d shape`推导方法可以[参考这里](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d)，`Conv2d`的`pad_mode`改成`same`，或者可以根据`Conv2d shape`推导公式自行计算`pad`，想要使得`shape`不变，一般pad为`(kernel_size-1)//2`。

<br/>

<font size=3>**Q：MindSpore安装完成，执行训练时发现网络性能异常，权重初始化耗时过长，怎么办？**</font>  

A：可能与环境中使用了`scipy 1.4`系列版本有关，通过`pip list | grep scipy`命令可查看scipy版本，建议改成MindSpore要求的`scipy`版本。版本第三方库依赖可以在`requirement.txt`中查看。
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> 其中version替换为MindSpore具体的版本分支。

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

<font size=3>**Q：请问自己制作的黑底白字`28*28`的数字图片，使用MindSpore训练出来的模型做预测，报错提示`wrong shape of image`是怎么回事？**</font>

A：首先MindSpore训练使用的灰度图MNIST数据集。所以模型使用时对数据是有要求的，需要设置为`28*28`的灰度图，就是单通道才可以。

<br/>

<font size=3>**Q：在Ascend平台上，执行用例有时候会报错`run task error`，如何获取更详细的日志帮助问题定位？**</font>

A：使用msnpureport工具设置device侧日志级别，工具位置在：`/usr/local/Ascend/driver/tools/msnpureport`。

- 全局级别：

```bash
/usr/local/Ascend/driver/tools/msnpureport -g info
```

- 模块级别：

```bash
/usr/local/Ascend/driver/tools/msnpureport -m SLOG:error
````

- Event级别：

```bash
/usr/local/Ascend/driver/tools/msnpureport -e disable/enable
```

- 多device id级别：

```bash
/usr/local/Ascend/driver/tools/msnpureport -d 1 -g warning
```

假设deviceID的取值范围是[0-7]，`device0`-`device3`和`device4`-`device7`分别在一个os上。其中`device0`-`device3`共用一个日志配置文件；`device4`-`device7`共用一个配置文件。如果修改了`device0`-`device3`中的任意一个日志级别，其他`device`的日志级别也会被修改。如果修改了`device4`-`device7`中的任意一个日志级别，其他device的日志级别也会被修改。

`Driver`包安装以后（假设安装路径为/usr/local/HiAI，在Windows环境下，`msnpureport.exe`执行文件在C:\ProgramFiles\Huawei\Ascend\Driver\tools\目录下），假设用户在/home/shihangbo/目录下直接执行命令行，则Device侧日志被导出到当前目录下，并以时间戳命名文件夹进行存放。

<br/>

<font size=3>**Q：使用ExpandDims算子报错：`Pynative run op ExpandDims failed`。具体代码：**</font>

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

用户可以参考[官网教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/debug_in_pynative_mode.html)选择合适、统一的模式和写法来完成训练。

<br/>

<font size=3>**Q：使用Ascend平台执行训练过程，出现报错：`Out of Memory!!! total[3212254720] (dynamic[0] memory poll[524288000]) malloc[32611480064] failed!` 如何解决？**</font>

A：此问题属于内存占用过多导致的内存不够问题，可能原因有两种：

- `batch_size`的值设置过大。解决办法：将`batch_size`的值设置减小。
- 引入了异常大的`Parameter`，例如单个数据shape为[640,1024,80,81]，数据类型为float32，单个数据大小超过15G，这样差不多大小的两个数据相加时，占用内存超过3*15G，容易造成`Out of Memory`。解决办法：检查参数的`shape`，如果异常过大，减少shape。
- 如果以上操作还是未能解决，可以上[官方论坛](https://bbs.huaweicloud.com/forum/forum-1076-1.html)发帖提出问题，将会有专门的技术人员帮助解决。

<br/>

<font size=3>**Q：MindSpore执行GPU分布式训练报错如下，如何解决：**</font>

```text
Loading libgpu_collective.so failed. Many reasons could cause this:
1.libgpu_collective.so is not installed.
2.nccl is not installed or found.
3.mpi is not installed or found
```

A：此问题为MindSpore动态加载集合通信库失败，可能原因如下：

- 执行环境未安装分布式训练依赖的OpenMPI以及NCCL。
- NCCL版本未更新至`v2.7.6`：MindSpore `v1.1.0`新增GPU P2P通信算子，该特性依赖于NCCL `v2.7.6`，若环境使用的NCCL未升级为此版本，则会引起加载失败错误。

<br/>

<font size=3>**Q：启动缓存服务器时，若提示找不到`libpython3.7m.so.1.0`文件，应如何处理？**</font>

A：尝试在虚拟环境下查找其路径并设置LD_LIBRARY_PATH变量：

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib
```

<br/>

<font size=3>**Q：缓存服务器异常关闭如何处理？**</font>

A：缓存服务器使用过程中，会进行IPC共享内存和socket文件等系统资源的分配。若允许溢出，在磁盘空间还会存在溢出的数据文件。一般情况下，如果通过`cache_admin --stop`命令正常关闭服务器，这些资源将会被自动清理。

但如果缓存服务器被异常关闭，例如缓存服务进程被杀等，用户需要首先尝试重新启动服务器，若启动失败，则应该依照以下步骤手动清理系统资源：

- 删除IPC资源。

    1. 检查是否有IPC共享内存残留。

        一般情况下，系统会为缓存服务分配4GB的共享内存。通过以下命令可以查看系统中的共享内存块使用情况。

        ```shell
        $ ipcs -m
        ------ Shared Memory Segments --------
        key        shmid      owner      perms      bytes      nattch     status
        0x61020024 15532037   root       666        4294967296 1
        ```

        其中，`shmid`为共享内存块id，`bytes`为共享内存块的大小，`nattch`为链接到该共享内存块的进程数量。`nattch`不为0表示仍有进程使用该共享内存块。在删除共享内存前，需要停止使用该内存块的所有进程。

    2. 删除IPC共享内存。

        找到对应的共享内存id，并通过以下命令删除。

        ```shell
        ipcrm -m {shmid}
        ```

- 删除socket文件。

    一般情况下，socket文件位于`/tmp/mindspore/cache`。进入文件夹，执行以下命令删除socket文件。

    ```shell
    rm cache_server_p{port_number}
    ```

    其中`port_number`为用户创建缓存服务器时指定的端口号，默认为50052。

- 删除溢出到磁盘空间的数据文件。

    进入启用缓存服务器时指定的溢出数据路径。通常，默认溢出路径为`/tmp/mindspore/cache`。找到路径下对应的数据文件夹并逐一删除。

<br/>

<font size=3>**Q：使用GPU版本MindSpore时，如何设置`DEVICE_ID`环境变量**</font>

A：MindSpore GPU模式一般无需设置`DEVICE_ID`环境变量，MindSpore会根据cuda环境变量`CUDA_VISIBLE_DEVICES`，自动选择可见的GPU设备。设置`CUDA_VISIBLE_DEVICES`环境变量后，则`DEVICE_ID`环境变量代表可见GPU设备的下标：

- 执行`export CUDA_VISIBLE_DEVICES=1,3,5`后，`DEVICE_ID`应当被设置为`0`，`1`或`2`，若设置为`3`及以上，MindSpore会由于设备ID不合法而运行失败。

<br/>

<font size=3>**Q：MindSpore代码里面的model_zoo/official/cv/resnet/train.py中context.set_ps_context(enable_ps=True)为什么一定要在init之前设置**</font>

A：MindSpore Ascend模式下，如果先调用init，那么会为所有的进程都分配卡，但是parameter server训练模式下server是不需要分配卡的，那么worker和server就会去使用同一块卡，导致会报错：Hccl dependent tsd is not open。

