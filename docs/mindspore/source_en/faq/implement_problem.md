# Implement Problem

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/implement_problem.md)

## Q: How do I use MindSpore to implement multi-scale training?

A: During multi-scale training, when different `shape` are used to call `Cell` objects, different graphs are automatically built and called based on different `shape`, to implement the multi-scale training. Note that multi-scale training supports only the non-data sink mode and does not support the data offloading mode. For details, see the multi-scale training implement of [yolov3](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3).

<br/>

## Q: If a `tensor` of MindSpore whose `requires_grad=False` is set to `False` is converted into `numpy` for processing and then converted into `tensor`, will the computational graph and backward propagation be affected?

A: In PyNative mode, if `numpy` is used for computation, gradient transfer will be interrupted. In the scenario where `requires_grad=False` is set to `False`, if the backward propagation of `tensor` is not transferred to other parameters, there is no impact. If `requires_grad=False` is set to `True`, there is an impact.

<br/>

## Q: How do I modify the `weight` and `bias` of the fully-connected layer like `torch.nn.functional.linear()`?

A: The `nn.Dense` interface is similar to `torch.nn.functional.linear()`. `nn.Dense` can specify the initial values of `weight` and `bias`. Subsequent changes are automatically updated by the optimizer. During the training, you do not need to change the values of the two parameters.

<br/>

## Q: What is the function of the `.meta` file generated after the model is saved using MindSpore? Can the `.meta` file be used to import the graph structure?

A: The `.meta` file is a compiled graph structure. However, this structure cannot be directly imported currently. If you do not know the graph structure, you still need to use the MindIR file to import the network.

<br/>

## Q: Can the `yolov4-tiny-3l.weights` model file be directly converted into a MindSpore model?

A: No. You need to convert the parameters trained by other frameworks into the MindSpore format, and then convert the model into a MindSpore model.

<br/>

## Q: Why an error message is displayed when MindSpore is used to set `model.train`?

```python
model.train(1, dataset, callbacks=ms.train.LossMonitor(1), dataset_sink_mode=True)
model.train(1, dataset, callbacks=ms.train.LossMonitor(1), dataset_sink_mode=False)
```

A: If the offloading mode has been set, it cannot be set to non-offloading mode, which is a restriction on the running mechanism.

<br/>

## Q: What should I pay attention to when using MindSpore to train a model in the `eval` phase? Can the network and parameters be loaded directly? Does the optimizer need to be used in the Model?

A: It mainly depends on what is required in the `eval` phase. For example, the output of the `eval` network of the image classification task is the probability of each class, and the `acc` is circulated with the corresponding label.
In most cases, the training network and parameters can be directly reused. Note that the inference mode needs to be set.

```python
net.set_train(False)
```

The optimizer is not required in the `eval` phase. However, if the `model.eval` API of MindSpore needs to be used, the `loss function` needs to be configured. For example:

```python
# Define a model.
model = ms.train.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
# Evaluate the model.
res = model.eval(dataset)
```

<br/>

## Q: How do I use `param_group` in SGD to reduce the learning rate?

A: To change the value according to `epoch`, use [Dynamic LR Function](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#dynamic-lr-function) and set `step_per_epoch` to `step_size`. To change the value according to `step`, set `step_per_epoch` to 1. You can also use [LearningRateSchedule](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#learningrateschedule-class).

<br/>

## Q: How do I modify parameters (such as the dropout value) on MindSpore?

A: When building a network, use `if self.training: x = dropput(x)`. When inferring, set `network.set_train(False)` before execution to disable the dropout function. During training, set `network.set_train(mode_false)` to True to enable the dropout function.

<br/>

## Q: How do I view the number of model parameters?

A: You can load the checkpoint count directly. Variables in the momentum and optimizer may be counted, so you need to filter them out.
You can refer to the following APIs to collect the number of network parameters:

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

[Script Link](https://gitee.com/mindspore/models/blob/master/research/cv/tinynet/src/utils.py).

<br/>

## Q: How do I monitor the `loss` during training and save the training parameters when the `loss` is the lowest?

A: You can customize the `Callback` method. Refer to the `ModelCheckpoint` writeup, and in addition add the logic to determine the `loss`.

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

## Q: How do I obtain  `feature map` with the expected size when `nn.Conv2d` is used?

A: For details about how to derive the `Conv2d shape`, click [here](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d). Change `pad_mode` of `Conv2d` to `same`. Alternatively, you can calculate the `pad` based on the `Conv2d shape` derivation formula to keep the `shape` unchanged. Generally, the pad is `(kernel_size-1)//2`.

<br/>

## Q: Can MindSpore be used to customize a loss function that can return multiple values?

A: After customizing the `loss function`, you need to customize `TrainOneStepCell`. The number of `sens` for implementing gradient calculation is the same as that of `network` outputs. For details, see the following:

```python
net = Net()
loss_fn = MyLoss()
loss_with_net = MyWithLossCell(net, loss_fn)
train_net = MyTrainOneStepCell(loss_with_net, optim)
model = ms.train.Model(net=train_net, loss_fn=None, optimizer=None)
```

<br/>

## Q: How does MindSpore implement the early stopping function?

A: You can refer to [EarlyStopping](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.EarlyStopping.html).

<br/>

## Q: After a model is trained, how do I save the model output in text or `npy` format?

A: The network output is `Tensor`. You need to use the `asnumpy()` method to convert the `Tensor` to `numpy` and then save the data. For details, see the following:

```python
out = net(x)
np.save("output.npy", out.asnumpy())
```

<br/>

## Q: How to handle cache server exception shutdown?

A: During the use of the cache server, system resources such as IPC share memory and socket files are allocated. If overflow is allowed, there will be overflowing data files on disk space. In general, if the server is shut down normally via the `dataset-cache --stop` command, these resources will be automatically cleaned up.

However, if the cache server is shut down abnormally, such as the cache service process is killed, the user needs to try to restart the server first. If the startup fails, you should follow the following steps to manually clean up the system resources:

- Delete the IPC resource.

    1. Check for IPC shared memory residue.

        In general, the system allocates 4GB of share memory for the caching service. The following command allows you to view the usage of share memory blocks in the system.

        ```text
        $ ipcs -m
        ------ Shared Memory Segments --------
        key        shmid      owner      perms      bytes      nattch     status
        0x61020024 15532037   root       666        4294967296 1
        ```

        where `shmid` is the share memory block id, `bytes` is the size of the share memory block, and `nattch` is the number of processes linking to the shared memory block.  `nattch` is not 0, which indicates that there are still processes that use the share memory block. Before you delete share memory, you need to stop all processes that use that memory block.

    2. Delete the IPC share memory.

        Find the corresponding share memory id, and delete via the following command.

        ```text
        ipcrm -m {shmid}
        ```

- Delete socket files.

    In general, socket files is located `/tmp/mindspore/cache`. Enter the folder, and execute the following command to delete socket files.

    ```text
    rm cache_server_p{port_number}
    ```

    where `port_number` is the port number specified when the user creates the cache server, which defaults to 50052.

- Delete data files that overflow to disk space.

    Enter the specified overflow data path when you enabled the cache server. In general, the default overflow path is `/tmp/mindspore/cache`. Find the corresponding data folders under the path and delete them one by one.

<br/>

## Q: Can the `vgg16` model be loaded by using the GPU via Hub and whether can the migration model be done?

A: Please manually modify the following two parameters:

```python
# Increase **kwargs parameter: as the following
def vgg16(num_classes=1000, args=None, phase="train", **kwargs):
```

```python
# Increase **kwargs parameter: as the following
net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase, **kwargs)
```

<br/>

## Q: How to obtain middle-layer features of a VGG model?

A: Obtaining the middle-layer features of a network is not closely related to the specific framework. For the `vgg` model defined in `torchvison`, the `features` field can be used to obtain the "middle-layer features". The `vgg` source code of `torchvison` is as follows:

```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
```

The `vgg16` defined in MindSpore can be obtained through the `layers` field as follows:

```python
network = vgg16()
print(network.layers)
```

<br/>

## Q: When MindSpore is used for model training, there are four input parameters for `CTCLoss`: `inputs`, `labels_indices`, `labels_values`, and `sequence_length`. How do I use `CTCLoss` for model training?

A: The `dataset` received by the defined `model.train` API can consist of multiple pieces of data, for example, (`data1`, `data2`, `data3`, ...). Therefore, the `dataset` can contain `inputs`, `labels_indices`, `labels_values`, and `sequence_length` information. You only need to define the dataset in the corresponding format and transfer it to `model.train`. For details, see [Data Processing API](https://www.mindspore.cn/docs/en/master/features/index.html).

<br/>

## Q: What are the available recommendation or text generation networks or models provided by MindSpore?

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore ModelZoo](https://gitee.com/mindspore/models/blob/master/README.md#).

<br/>

## Q: How do I execute a single `ut` case in `mindspore/tests`?

A: `ut` cases are usually based on the MindSpore package of the debug version, which is not provided on the official website. You can run `sh build.sh` to compile based on the source code and then run the `pytest` command. The compilation in debug mode does not depend on the backend. Compile the `sh build.sh -t on` option. For details about how to execute cases, see the `tests/runtest.sh` script.

<br/>

## Q: For Ascend users, how to get more detailed logs to help position the problems when the `run task error` is reported during executing the cases?

A: Use the msnpureport tool to set the on-device log level. The tool is stored in `/usr/local/Ascend/latest/driver/tools/msnpureport`.

- Global-level:

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -g info
```

- Module-level

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -m SLOG:error
```

- Event-level

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -e disable/enable
```

- Multi-device ID-level

```bash
/usr/local/Ascend/latest/driver/tools/msnpureport -d 1 -g warning
```

Assume that the value range of deviceID is [0, 7], and `devices 0-3` and `devices 4-7` are on the same OS. `devices 0` to `device3` share the same log configuration file and `device4` to `device7` share the same configuration file. In this way, changing any log level in `devices 0` to `device3` will change that of other `device`. This rule also applies to `device4` to `device7` .

After the `Driver` package is installed (assuming that the installation path is /usr/local/HiAI and the execution file `msnpureport.exe` is in the C:\ProgramFiles\Huawei\Ascend\Driver\tools\ directory on Windows), suppose the user executes the command line directly in the /home/shihangbo/directory, the Device side logs are exported to the current directory and stored in a timestamp-named folder.

<br/>

## Q: How can I do when the error message `Out of Memory!!! total[3212254720] (dynamic[0] memory poll[524288000]) malloc[32611480064] failed!` is displayed by performing the training process using the Ascend platform?

A: This issue is a memory shortage problem caused by too much memory usage, which can be caused by two possible causes:

- Set the value of `batch_size` too large. Solution: Reduce the value of `batch_size`.
- Introduce the abnormally large `parameter`, for example, a single data shape is [640,1024,80,81]. The data type is  float32, and the single data size is over 15G. In this way, the two data with the similar size are added together, and the memory occupied is over 3*15G, which easily causes `Out of Memory`. Solution: Check the `shape` of the parameter. If it is abnormally large, the shape can be reduced.
- If the following operations cannot solve the problem, you can raise the problem on the [official forum](https://www.hiascend.com/forum/forum-0106101385921175002-1.html), and there are dedicated technical personnels for help.

<br/>

## Q: How do I change hyperparameters for calculating loss values during neural network training?

A: Sorry, this function is not available yet. You can find the optimal hyperparameters by training, redefining an optimizer, and then training.

<br/>

## Q: What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` is displayed during application running?

A: While installing Atlas 200/300/500 inference product software packages depended by MindSpore, the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

<br/>

## Q: Why does set_ps_context(enable_ps=True) in model_zoo/official/cv/ResNet/train.py in the MindSpore code have to be set before init?

A: In MindSpore Ascend mode, if init is called first, all processes will be allocated cards, but in parameter server training mode, the server does not need to allocate cards, and the worker and server will use the same card, resulting in an error: Ascend kernel runtime initialization failed.

<br/>

## Q: What should I do when an error `Stream isn't enough` is displayed during executing the model on the Ascend platform?

A: When resnet50 training is performed on the CPU ARM, some operators are implemented based on the oneDNN library, and the oneDNN library achieves multi-threaded parallelism based on the libgomp library. Currently, there is a problem in libgomp where the number of threads configured for multiple parallel domains is different and the memory consumption continues to grow. The continuous growth of the memory can be controlled by configuring a uniform number of threads globally. For comprehensive performance considerations, it is recommended to configure a unified configuration to 1/4 of the number of physical cores, such as `export OMP_NUM_THREADS=32`.

<br/>

## Q: What should I do when an error `Stream isn't enough` is displayed during executing the model on the Ascend platform?

A: Stream represents an operation queue. Tasks on the same stream are executed in sequence, and different streams can be executed in parallel. Various operations in the network generate tasks and are assigned to streams to control the concurrent mode of task execution. Ascend platform has a limit on the number of tasks on the same stream, and tasks that exceed the limit will be assigned to new streams. The multiple parallel methods of MindSpore will also be assigned to new streams, such as parallel communication operators. Therefore, when the number of assigned streams exceeds the resource limit of the Ascend platform, an error will be reported. Reference solution:

- Reduce the size of the network model

- Reduce the use of communication operators in the network

- Reduce conditional control statements in the network

<br/>

## Q: On the Ascend platform, if an error "Ascend error occurred, error message:" is reported in the log and followed by an error code, such as "E40011", how to find the cause of the error code?

A: When "Ascend error occurred, error message:" appears, it indicates that a module of Ascend CANN is abnormal and the error log is reported.

At this time, there is an error message after the error code. If you need a more detailed possible cause and solution for this exception, please refer to the "error code introducing" section of the corresponding Ascend version document, such as [CANN canncommercial version 7.0.0 Error Code introducing](https://www.hiascend.com/document/detail/zh/canncommercial/700/troublemanagement/troubleshooting/atlaserrorcode_15_0139.html).

<br/>

## Q: When the third-party component gensim is used to train the NLP network, the error "ValueError" may be reported. What can I do?

A: The following error information is displayed:

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

For details about the error cause, see the [gensim](https://github.com/RaRe-Technologies/gensim/issues/3095) or [numpy](https://github.com/numpy/numpy/issues/18709) official website.

Solutions:

Method 1: Reinstall the Numpy and Gensim and run the following commands: `pip uninstall gensim numpy -y && pip install numpy==1.18.5 gensim`

Method 2: If the problem persists, delete the cache file of the wheel installation package and then perform method 1. (The cache directory of the wheel installation package is `~/.cache/pip/wheels`)

<br/>

## Q: What should I do if I encounter `matplotlib.pyplot.show()` or `plt.show` not be executed during the documentation sample code is running?

A: First confirm whether `matplotlib` is installed. If it is not installed, you can execute `pip install matplotlib` on the command line to install it.

Secondly, because the function of `matplotlib.pyplot.show()` is to display graph data graphically, it is necessary to run the system to support the graph display function. If the system cannot support graph display, the reader needs to comment out the command line of the graph display. Operation will not affect the results of the overall code.

<br/>

## Q: How to handle running failures when encountering an online runtime provided in the documentation?

A: Need to confirm that the following preparations have been done.

- First, you need to log in to ModelArts through your HUAWEI CLOUD account.
- Secondly, note that the hardware environment supported by the tags in the tutorial document and the hardware environment configured in the example code is Ascend, GPU or CPU. Since the hardware environment used by default after login is CPU, the Ascend environment and GPU environment need to be switched manually by the user.
- Finally, confirm that the current `Kernel` is MindSpore.

After completing the above steps, you can run the tutorial.

For the specific operation process, please refer to [Based on ModelArts Online Experience MindSpore](https://www.hiascend.com/developer/blog/details/0254122007639293043).

<br/>

## Q: No error is reported when using result of division in GRAPH mode, but an error is reported when using result of division in PYNATIVE mode?

A: In GRAPH mode, since the graph compilation is used, the data type of the output result of the operator is determined at the graph compilation stage.

For example, the following code is executed in GRAPH mode, and the type of input data is int, so the output result is also int type according to graph compilation.

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

output:

```text
4 <class 'int'>
```

Change the execution mode and change GRAPH_MODE to PYNATIVE_MODE. Since the Python syntax is used in PyNative mode, the type of any division output to Python syntax is float type, so the execution result is as follows.

```text
4.0 <class 'float'>
```

Therefore, in the scenario where the subsequent operator clearly needs to use int, it is recommended to use Python's divisibility symbol `//`.

<br/>

## Q: What can I do when the error message `MemoryError: std::bad_alloc` is reported during the execution of the operator?

A: The reason for this error is that the user did not configure the operator parameters correctly, so that the memory space applied by the operator exceeded the system memory limit, and the system failed to allocate memory. The following uses mindspore.ops.UniformCandidateSampler as an example for description.

- UniformCandidateSampler samples a set of classes by using uniform distribution. According to the parameter `num_sampled` set by the user, the shape of output tensor would be `(num_sampled,)`.

- When the user sets `num_sampled=int64.max`, the memory space requested by the output tensor exceeds the system memory limit, causing `bad_alloc`.

Therefore, the user needs to set the operator parameters appropriately to avoid such errors.

<br/>

## Q: How do I understand the "Ascend Error Message" in the error message?

A: The "Ascend Error Message" is a fault message thrown after there is an error during CANN execution when CANN (Ascend Heterogeneous Computing Architecture) interface is called by MindSpore, which contains information such as error code and error description. For example:

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
 EJ0001: Failed to initialize the HCCP process. Reason: Maybe the last training process is running. //EJ0001 is the error code, followed by the description and cause of the error. The cause of the error in this example is that the distributed training of the same 8 nodes was started several times, causing process conflicts
 Solution: Wait for 10s after killing the last training process and try again. //The print message here gives the solution to the problem, and this example suggests that the user clean up the process
 TraceBack (most recent call last): //The information printed here is the stack information used by the developer for positioning, and generally the user do not need to pay attention
```

```text
tsd client wait response fail, device response code[1]. unknown device  error.[FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:233]
```

In addition, CANN may throw some Inner Errors, for example, the error code is "EI9999: Inner Error". If you cannot search the case description in MindSpore official website or forum, you can ask for help in the community by raising an issue.

<br/>

## Q: How to control the Tensor value printed by the `print` method?

A: In PyNative dynamic graph mode, you can use numpy native methods such as ` set_ Printoptions ` to control the output value. In the Graph static graph mode, because the `print` method needs to be converted into an operator, the output value cannot be controlled temporarily. For specific usage of print operator, see [Reference](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Print.html).
<br/>

## Q: How does `Tensor.asnumpy()` share the underlying storage with Tensor?

A: `Tensor.asnumpy()` will convert the Tensor to a NumPy ndarray. This tensor and the returned ndarray by `Tensor.asnumpy()` share the same underlying storage on the host side. On the host side, changes to Tensor will be reflected in the ndarray and vice versa. It should be noted that changes on the host side cannot be automatically synchronized to the device side. For example:

```text
import mindspore as ms
x = ms.Tensor([1, 2, 3]) + ms.Tensor([4, 5, 6])
y = x.asnumpy()

# x is the result of operation calculation on the device side, and y is on the host side.
# The changes of y on the host side cannot be automatically synchronized to x on the device side.
y[0] = 11
print(y)

# Printing x triggers a data sync, which syncs the data of x to y.
print(x)
print(y)
```

The result is as follows:

```text
[11 7 9]
[5 7 9]
[5 7 9]
```

<br/>

## Q: Why will running the script on GPU stuck for a long time on version 1.8?

A: In order to be compatible with more GPU architectures, NVCC compiles CUDA files into PTX files first, and compiles them into binary executable files when using them for the first time. Therefore, compilation time will be consumed.
Compared with the previous version, version 1.8 has added many CUDA operators, resulting in an increase in the compilation time of this part (The time varies according to the equipment. For example, the first compilation time on V100 is about 5 minutes).
This compilation will generate a cache file (taking the Ubuntu system as an example, the cache file is located in `~/.nv/ComputeCache`), and the cache file will be directly loaded during subsequent execution.
Therefore, it will be stuck for several minutes during the first use, and the subsequent use will be a normal time consumption.

Subsequent versions will be pre-compiled and optimized.

<br/>
