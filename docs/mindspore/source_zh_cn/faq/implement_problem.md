# 执行问题

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/implement_problem.md)

## Q: 请问使用MindSpore如何实现多尺度训练？

A: 在多尺度训练过程中，使用不同`shape`调用`Cell`对象的时候，会自动根据不同`shape`编译并调用不同的图，从而实现多尺度的训练。要注意多尺度训练只支持非数据下沉模式，不能支持数据下沉的训练方式。可以参考[yolov3](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3)的多尺度训练实现。

<br/>

## Q: 如果MindSpore的`requires_grad=False`的`tensor`转化为`numpy`类型进行处理然后再转化会`tensor`，会对计算图和反向传播有影响吗？

A: 在PyNative模式下，如果中间使用`numpy`计算，会导致梯度传递中断，`requires_grad=False`的场景下，如果该`tensor`的反向传播不传给其他参数使用，是没有影响的；如果`requires_grad=True`的场景下，是有影响的。

<br/>

## Q: 请问怎样实现类似`torch.nn.functional.linear()`那样能够对全连接层`weight`、`bias`进行修改，应该如何操作？

A: MindSpore与`torch.nn.functional.linear()`功能最接近的接口就是`nn.Dense`了。`nn.Dense`能指定`weight`和`bias`的初始值，后续的变化是由优化器自动更新的。训练过程中，用户不需要主动修改这两个参数的值。

<br/>

## Q: 使用MindSpore在模型保存后生成的`.meta`文件作用是什么，可以用`.meta`文件导入图结构吗？

A: 这里的`.meta`文件是编译好的图结构，但是目前并不支持直接导入这种结构。如果不知道图结构的情况下想要导入网络，还是需要用MindIR格式的文件。

<br/>

## Q: 请问`yolov4-tiny-3l.weights`模型文件可以直接转换成MindSpore模型吗？

A: 不能的，需要把其他框架训练好的参数转换成MindSpore的格式，才能转成MindSpore的模型。

<br/>

## Q: 使用MindSpore进行`model.train`的时候进行了如下设置，为什么会报错呢？

```python
model.train(1, dataset, callbacks=ms.train.LossMonitor(1), dataset_sink_mode=True)
model.train(1, dataset, callbacks=ms.train.LossMonitor(1), dataset_sink_mode=False)
```

A: 因为在已经设置为下沉模式的情况下，就不能再设置为非下沉了，是运行机制上的限制。

<br/>

## Q: 使用MindSpore训练模型在`eval`阶段，需要注意什么？能够直接加载网络和参数吗？需要在Model中使用优化器吗？

A: 在`eval`阶段主要看需要什么，比如图像分类任务`eval`网络的输出是各个类的概率值，与对应标签计算`acc`。
大多数情况是可以直接复用训练的网络和参数的，需要注意的是需要设置推理模式。

```python
net.set_train(False)
```

在eval阶段不需要优化器，但是需要使用MindSpore的`model.eval`接口的话需要配置一下`loss function`，如：

```python
# 定义模型
model = ms.train.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
# 评估模型
res = model.eval(dataset)
```

<br/>

## Q: 如何使用SGD里的`param_group`来实现学习率的衰减？

A: 如果需要按照`epoch`来变化，可以使用[Dynamic LR](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#dynamic-lr函数),把其中的`step_per_epoch`设置成`step_size`，如果需要按照`step`来变化，可以把其中的`step_per_epoch`设置成1，也可以用[LearningRateSchedule](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#learningrateschedule类)。

<br/>

## Q: MindSpore如何进行参数（如dropout值）修改？

A: 在构造网络的时候可以通过 `if self.training: x = dropput(x)`，推理时，执行前设置`network.set_train(False)`，就可以不使用dropout，训练时设置为True就可以使用dropout。

<br/>

## Q: 如何查看模型参数量？

A: 可以直接加载CheckPoint统计，可能额外统计了动量和optimizer中的变量，需要过滤下相关变量。
您可以参考如下接口统计网络参数量:

```python
def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): MindSpore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params
```

具体[脚本链接](https://gitee.com/mindspore/models/blob/master/research/cv/tinynet/src/utils.py)。

<br/>

## Q: 如何在训练过程中监控`loss`在最低的时候并保存训练参数？

A: 可以自定义一个`Callback`。参考`ModelCheckpoint`的写法，此外再增加判断`loss`的逻辑:

```python
class EarlyStop(Callback):
    def __init__(self, control_loss=1):
        super(EarlyStop, self).__init__()
        self._control_loss = control_loss

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if loss.asnumpy() < self._control_loss:
            # Stop training
            run_context._stop_requested = True

stop_cb = EarlyStop(control_loss=1)
model.train(epoch_size, ds_train, callbacks=[stop_cb])
```

<br/>

## Q: 使用`nn.Conv2d`时，怎样获取期望大小的`feature map`？

A: `Conv2d shape`推导方法可以[参考这里](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d)，`Conv2d`的`pad_mode`改成`same`，或者可以根据`Conv2d shape`推导公式自行计算`pad`，想要使得`shape`不变，一般pad为`(kernel_size-1)//2`。

<br/>

## Q: 使用MindSpore可以自定义一个可以返回多个值的loss函数？

A: 自定义`loss function`后还需自定义`TrainOneStepCell`，实现梯度计算时`sens`的个数和`network`的输出个数相同。具体可参考:

```python
net = Net()
loss_fn = MyLoss()
loss_with_net = MyWithLossCell(net, loss_fn)
train_net = MyTrainOneStepCell(loss_with_net, optim)
model = ms.train.Model(net=train_net, loss_fn=None, optimizer=None)
```

<br/>

## Q: MindSpore如何实现早停功能？

A：可以使用[EarlyStopping 方法](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.EarlyStopping.html)。

<br/>

## Q: 模型已经训练好，如何将模型的输出结果保存为文本或者`npy`的格式？

A: 您好，我们网络的输出为`Tensor`，需要使用`asnumpy()`方法将`Tensor`转换为`numpy`，再进行下一步保存。具体可参考:

```python
out = net(x)
np.save("output.npy", out.asnumpy())
```

<br/>

## Q: 缓存服务器异常关闭如何处理？

A: 缓存服务器使用过程中，会进行IPC共享内存和socket文件等系统资源的分配。若允许溢出，在磁盘空间还会存在溢出的数据文件。一般情况下，如果通过`dataset-cache --stop`命令正常关闭服务器，这些资源将会被自动清理。

但如果缓存服务器被异常关闭，例如缓存服务进程被杀等，用户需要首先尝试重新启动服务器，若启动失败，则应该依照以下步骤手动清理系统资源:

- 删除IPC资源。

    1. 检查是否有IPC共享内存残留。

        一般情况下，系统会为缓存服务分配4GB的共享内存。通过以下命令可以查看系统中的共享内存块使用情况。

        ```text
        $ ipcs -m
        ------ Shared Memory Segments --------
        key        shmid      owner      perms      bytes      nattch     status
        0x61020024 15532037   root       666        4294967296 1
        ```

        其中，`shmid`为共享内存块id，`bytes`为共享内存块的大小，`nattch`为链接到该共享内存块的进程数量。`nattch`不为0表示仍有进程使用该共享内存块。在删除共享内存前，需要停止使用该内存块的所有进程。

    2. 删除IPC共享内存。

        找到对应的共享内存id，并通过以下命令删除。

        ```text
        ipcrm -m {shmid}
        ```

- 删除socket文件。

    一般情况下，socket文件位于`/tmp/mindspore/cache`。进入文件夹，执行以下命令删除socket文件。

    ```text
    rm cache_server_p{port_number}
    ```

    其中`port_number`为用户创建缓存服务器时指定的端口号，默认为50052。

- 删除溢出到磁盘空间的数据文件。

    进入启用缓存服务器时指定的溢出数据路径。通常，默认溢出路径为`/tmp/mindspore/cache`。找到路径下对应的数据文件夹并逐一删除。

<br/>

## Q: 通过Hub可以使用GPU加载`vgg16`模型以及是否可以做迁移模型吗？

A: 请手动修改如下两处参数即可:

```python
# 增加**kwargs参数: 如下
def vgg16(num_classes=1000, args=None, phase="train", **kwargs):
```

```python
# 增加**kwargs参数: 如下
net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase, **kwargs)
```

<br/>

## Q: 如何得到VGG模型中间层特征？

A: 你好，获取网络中间层的特征，其实跟具体框架没有太大关系了。`torchvison`里定义的`vgg`模型，可以通过`features`字段获取"中间层特征"，`torchvison`的`vgg`源码如下:

```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
```

在MindSpore里定义的`vgg16`，可以通过`layers`字段获取，如下:

```python
network = vgg16()
print(network.layers)
```

<br/>

## Q: 使用MindSpore进行模型训练时，`CTCLoss`的输入参数有四个: `inputs`、`labels_indices`、`labels_values`、`sequence_length`，如何使用`CTCLoss`进行训练？

A: 定义的`model.train`接口里接收的`dataset`可以是多个数据组成，形如(`data1`、`data2`、`data3`...)，所以`dataset`是可以包含`inputs`、`labels_indices`、`labels_values`、`sequence_length`的信息的。只需要定义好相应形式的`dataset`，传入`model.train`里就可以。具体的可以了解下相应的[数据处理接口](https://www.mindspore.cn/docs/zh-CN/master/features/index.html)。

<br/>

## Q: MindSpore有哪些现成的推荐类或生成类网络或模型可用？

A: 目前正在开发Wide & Deep、DeepFM、NCF等推荐类模型，NLP领域已经支持Bert_NEZHA，正在开发MASS等模型，用户可根据场景需要改造为生成类网络，可以关注[MindSpore ModelZoo](https://gitee.com/mindspore/models/blob/master/README_CN.md#)。

<br/>

## Q: `mindspore/tests`下怎样执行单个`ut`用例？

A: `ut`用例通常需要基于debug版本的MindSpore包，官网并没有提供。可以基于源码使用`sh build.sh`编译，然后通过`pytest`指令执行，debug模式编包不依赖后端。编译选项`sh build.sh -t on`，用例执行可以参考`tests/runtest.sh`脚本。

<br/>

## Q: 在Ascend平台上，执行用例有时候会报错`run task error`，如何获取更详细的日志帮助问题定位？

A: 使用msnpureport工具设置device侧日志级别，工具位置在: `/usr/local/Ascend/latest/driver/tools/msnpureport`。

- 全局级别:

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -g info
```

- 模块级别:

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -m SLOG:error
```

- Event级别:

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -e disable/enable
```

- 多device id级别:

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -d 1 -g warning
```

假设deviceID的取值范围是[0-7]，`device0`-`device3`和`device4`-`device7`分别在一个os上。其中`device0`-`device3`共用一个日志配置文件；`device4`-`device7`共用一个配置文件。如果修改了`device0`-`device3`中的任意一个日志级别，其他`device`的日志级别也会被修改。如果修改了`device4`-`device7`中的任意一个日志级别，其他device的日志级别也会被修改。

`Driver`包安装以后（假设安装路径为/usr/local/HiAI，在Windows环境下，`msnpureport.exe`执行文件在C:\ProgramFiles\Huawei\Ascend\Driver\tools\目录下），假设用户在/home/shihangbo/目录下直接执行命令行，则Device侧日志被导出到当前目录下，并以时间戳命名文件夹进行存放。

<br/>

## Q: 使用Ascend平台执行训练过程，出现报错: `Out of Memory!!! total[3212254720] (dynamic[0] memory poll[524288000]) malloc[32611480064] failed!` 如何解决？

A: 此问题属于内存占用过多导致的内存不够问题，可能原因有两种:

- `batch_size`的值设置过大。解决办法: 将`batch_size`的值设置减小。
- 引入了异常大的`Parameter`，例如单个数据shape为[640,1024,80,81]，数据类型为float32，单个数据大小超过15G，这样差不多大小的两个数据相加时，占用内存超过3*15G，容易造成`Out of Memory`。解决办法: 检查参数的`shape`，如果异常过大，减少shape。
- 如果以上操作还是未能解决，可以上[官方论坛](https://discuss.mindspore.cn/)发帖提出问题，将会有专门的技术人员帮助解决。

<br/>

## Q: 如何在训练神经网络过程中对计算损失的超参数进行改变？

A: 您好，很抱歉暂时还未有这样的功能。目前只能通过训练 -> 重新定义优化器 -> 训练，这样的过程寻找较优的超参数。

<br/>

## Q: 运行应用时报错`error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory`怎么办？

A: 安装MindSpore所依赖的Atlas 200/300/500推理产品配套软件包时，`CANN`包不能安装`nnrt`版本，而是需要安装功能完整的`toolkit`版本。

<br/>

## Q: MindSpore代码里面的model_zoo/official/cv/ResNet/train.py中set_ps_context(enable_ps=True)为什么一定要在init之前设置？

A: MindSpore Ascend模式下，如果先调用init，那么会为所有的进程都分配卡，但是parameter server训练模式下server是不需要分配卡的，那么worker和server就会去使用同一块卡，导致会报错: Ascend kernel runtime initialization failed。

<br/>

## Q: 在CPU ARM平台上进行resnet50训练，内存持续增长怎么办？

A: 在CPU ARM上进行resnet50训练时，部分算子的实现是基于oneDNN库，oneDNN库中是基于libgomp库实现多线程并行，当前libgomp存在多个并行域配置的线程数不同时有内存占用持续增长的问题。可通过全局配置统一的线程数来控制内存的持续增长。再综合性能上的考虑，建议统一配置为物理核数的1/4，比如`export OMP_NUM_THREADS=32`。

<br/>

## Q: 为什么在Ascend平台执行模型时报错`Stream isn't enough`？

A: 流表示一个操作队列，同一条流上的任务按序串行执行，不同流之间可以并行执行。网络中的各种操作会生成Task并被分配到流上，以控制任务执行的并发方式。由于Ascend平台对同一条流上的的任务数存在限制，超限的任务会分配新流，且MindSpore框架的多种并行方式也会分配新流，例如通信算子并行，因此当分配流的数目超过Ascend平台的资源限制就会报流超限的错误。参考解决方案：

- 减小网络模型规模

- 减少网络中通信算子的使用

- 减少网络中的条件控制语句

<br/>

## Q: 在Ascend平台上，日志中出现报错“Ascend error occurred, error message:”且跟随了一个错误码，如“E40011”，如何查找出现错误码的原因？

A: 当出现“Ascend error occurred, error message:”时，说明昇腾CANN相关模块出现异常，上报了错误日志。

此时错误码后有异常的错误信息。如果需要该异常更详细的可能原因和处理方法，请参考对应昇腾版本文档的《Error Code介绍》部分，如[昇腾CANN商用版(7.0.0) Error Code介绍](https://www.hiascend.com/document/detail/zh/canncommercial/700/troublemanagement/troubleshooting/atlaserrorcode_15_0139.html)。

<br/>

## Q: 训练nlp类网络，当使用第三方组件gensim时，可能会报错: ValueError，如何解决？

A: 以下为报错信息:

```bash
>>> import gensim
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/__init__.py", line 11, in <module>
    from gensim import parsing, corpora, matutils, interfaces, models, similarities, utils  # noqa:F401
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/corpora/__init__.py", line 6, in <module>
    from .indexedcorpus import IndexedCorpus  # noqa:F401 must appear before the other classes
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/corpora/indexedcorpus.py", line 14, in <module>
    from gensim import interfaces, utils
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/interfaces.py", line 19, in <module>
    from gensim import utils, matutils
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/matutils.py", line 1024, in <module>
    from gensim._matutils import logsumexp, mean_absolute_difference, dirichlet_expectation
  File "gensim/_matutils.pyx", line 1, in init gensim._matutils
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```

报错原因请参考[gensim](https://github.com/RaRe-Technologies/gensim/issues/3095)官网，或者[numpy](https://github.com/numpy/numpy/issues/18709)官网:

解决方案:

方法一: 重新安装numpy及gensim，执行命令: `pip uninstall gensim numpy -y && pip install numpy gensim` ；

方法二: 如果还是有问题，请删除wheel安装包的缓存文件，然后执行方法一（wheel安装包缓存目录为: `~/.cache/pip/wheels`）。

<br/>

## Q：运行文档示例代码的过程中，遇到`matplotlib.pyplot.show()`或`plt.show()`无法执行怎么处理？

A: 首先确认是否安装`matplotlib`，如果没有安装，可以在命令行中执行`pip install matplotlib`进行安装。

其次由于`matplotlib.pyplot.show()`的作用是以图形化方式展示，所以需要运行系统支持图形展示功能，如果系统不能支持图形展示，需要将该图形展示的命令行注释后再运行，不影响整体代码的运行结果。

<br/>

## Q: 使用文档中提供的在线运行时，遇到运行失败该如何处理？

A: 需要确认有做以下准备工作。

- 首先，需要通过华为云账号登录ModelArts。
- 其次，注意教程文档的标签中列举的硬件环境，以及样例代码中配置的硬件环境，是Ascend、GPU还是CPU，由于登录后默认使用的硬件环境是CPU，Ascend环境和GPU环境需要用户手动点击切换。
- 最后，确保当前`Kernel`为MindSpore。

完成上述步骤后，就可以运行文档了。

具体的操作过程可以参考[基于ModelArts在线体验MindSpore](https://www.hiascend.com/developer/blog/details/0254122007639293043)。

<br/>

## Q: 静态图下使用除法结果未报错，动态图下使用除法结果却报错？

A: 在静态图模式下，由于使用的是静态编译，对于算子输出结果的数据类型是在图编译阶段确定的。

例如如下代码在静态图模式下执行，输入数据的类型都为int类型，根据静态图编译，其输出结果也是int类型。

```python
import mindspore as ms
from mindspore import nn

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")

class MyTest(nn.Cell):
    def __init__(self):
        super(MyTest, self).__init__()

    def construct(self, x, y):
        return x / y
x = 16
y = 4
net = MyTest()
output = net(x, y)
print(output, type(output))
```

输出结果：

```text
4 <class 'int'>
```

修改执行模式，将GRAPH_MODE修改成PYNATIVE_MODE，由于在动态图模式下使用的Python语法执行，Python语法对任意除法输出的类型都是float类型，因此执行结果如下：

```text
4.0 <class 'float'>
```

因此在后续算子明确需要使用int的场景下，建议使用Python的整除符号`//`。

<br/>

## Q: 算子执行过程中出现报错: `MemoryError: std::bad_alloc` 如何解决？

A: 此问题的原因为：用户未正确配置算子参数，导致算子申请的内存空间超过了系统内存限制，进而系统分配内存失败。下面以算子 mindspore.ops.UniformCandidateSampler 为例进行说明：

- UniformCandidateSampler使用均匀分布对一组类别进行采样，根据用户设定的参数`num_sampled`，其输出Tensor的shape为`(num_sampled,)`。

- 当用户设定的`num_sampled=int64.max`时，其输出Tensor申请的内存空间超过了系统内存限制，并导致`bad_alloc`。

因此，用户需要适当设置算子参数，以避免此类报错。

<br/>

## Q: 如何理解报错提示中的"Ascend Error Message"？

 A: "Ascend Error Message"是MindSpore调用CANN(昇腾异构计算架构)接口时，CANN执行出错后抛出的故障信息，其中包含错误码和错误描述等信息，如下例子：

```text
Traceback (most recent call last):
 File "train.py", line 292, in <module>
 train_net()
 File  "/home/resnet_csj2/scripts/train_parallel0/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
 run_func(*args, **kwargs)
 File "train.py", line 227, in train_net
 set_parameter()
 File "train.py", line 114, in set_parameter
 init()
 File "/home/miniconda3/envs/ms/lib/python3.7/site-packages/mindspore/communication/management.py", line 149, in init
 init_hccl()
 RuntimeError: Ascend kernel runtime initialization failed.

 \----------------------------------------------------
 \- Ascend Error Message:
 \----------------------------------------------------
 EJ0001: Failed to initialize the HCCP process. Reason: Maybe the last training process is running. //EJ0001为错误码，之后是错误的描述与原因，本例子的错误原因是多次启动了相同8节点的分布式训练，造成进程冲突
 Solution: Wait for 10s after killing the last training process and try again. //此处打印信息给出了问题的解决方案，此例子建议用户清理进程
 TraceBack (most recent call last): //此处打印的信息是开发用于定位的堆栈信息，一般情况下用户不需关注
```

```text
tsd client wait response fail, device response code[1]. unknown device  error.[FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:233]
```

另外在一些情况下，CANN会抛出一些内部错误(Inner Error)，例如：错误码为 "EI9999: Inner Error" 此种情况如果在MindSpore官网或者论坛无法搜索到案例说明，可在社区提单求助。

<br/>

## Q: 如何控制`print`方法打印出的Tensor值？

A: 在PyNative动态图模式下，可以使用numpy原生方法如`set_printoptions`对输出的值进行控制。在Graph静态图模式下，因为`print`方法需要转化成为算子，所以暂时无法对输出的值进行控制。print算子具体用法可[参考](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Print.html)。

<br/>

## Q: `Tensor.asnumpy()`是怎么和Tensor共享内存地址的？

A: `Tensor.asnumpy()`会将Tensor本身转换为NumPy的ndarray。这个Tensor和`Tensor.asnumpy()`返回的ndarray共享host侧的内存地址，在host侧，对Tensor本身的修改会反映到相应的ndarray上，反之亦然。需要注意的是，host侧的修改无法自动同步到device侧。如：

```text
import mindspore as ms
x = ms.Tensor([1, 2, 3]) + ms.Tensor([4, 5, 6])
y = x.asnumpy()

# x 是 device 侧算子计算的结果，而 y 在 host 侧。host 侧对 y 的修改无法自动同步到 device 侧的 x。
y[0] = 11
print(y)

# 打印 x 会触发数据同步，将 x 的数据同步到 y。
print(x)
print(y)
```

运行结果如下：

```text
[11 7 9]
[5 7 9]
[5 7 9]
```

<br/>

## Q: 1.8版本首次运行GPU的脚本会卡很久？

A: 由于NVCC编译CUDA算子时为了兼容更多GPU架构，先编译成ptx文件，在首次使用时会进行JIT编译成二进制执行文件，因此会产生编译耗时。
而1.8版本相较于前版本增加了许多CUDA算子，导致这部分编译时间增加（不同设备时间不同，如在V100上首次编译时间为5分钟左右）。
该编译会产生缓存文件（以ubuntu系统为例，缓存文件位于 `~/.nv/ComputeCache` 路径下），在后续执行时会直接加载缓存文件。
因此会产生首次使用时会卡住几分钟，后续使用时间正常的现象。

后续版本会进行预编译优化。

<br/>
