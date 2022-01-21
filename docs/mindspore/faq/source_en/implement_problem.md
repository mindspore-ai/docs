# Implement Problem

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/faq/source_en/implement_problem.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: How do I use MindSpore to implement multi-scale training?**</font>

A: During multi-scale training, when different `shape` are used to call `Cell` objects, different graphs are automatically built and called based on different `shape`. Note that multi-scale training supports only the non-data sink mode and does not support the data offloading mode. For details, see the multi-scale training of the [yolov3](https://gitee.com/mindspore/models/tree/master/official/cv/yolov3_darknet53).

<br/>

<font size=3>**Q: If a `tensor` whose `requirements_grad` is set to `False` is converted into `numpy` for processing and then converted into `tensor`, will the computational graph and backward propagation be affected?**</font>

A: In PyNative mode, if `numpy` is used for computation, gradient transfer will be interrupted. In the scenario where `requirements_grad` is set to `False`, if the backward propagation of `tensor` is not transferred to other parameters, there is no impact. If `requirements_grad` is set to `True`, there is an impact.

<br/>

<font size=3>**Q: How do I modify the `weight` and `bias` of the fully-connected layer like `torch.nn.functional.linear()`?**</font>

A: The `nn.Dense` interface is similar to `torch.nn.functional.linear()`. `nn.Dense` can specify the initial values of `weight` and `bias`. Subsequent changes are automatically updated by the optimizer. During the training, you do not need to change the values of the two parameters.

<br/>

<font size=3>**Q: What is the function of the `.meta` file generated after the model is saved using MindSpore? Can the `.meta` file be used to import the graph structure?**</font>

A: The `.meta` file is a built graph structure. However, this structure cannot be directly imported currently. If you do not know the graph structure, you still need to use the MindIR file to import the network.

<br/>

<font size=3>**Q: Can the `yolov4-tiny-3l.weights` model file be directly converted into a MindSpore model?**</font>

A: No. You need to convert the parameters trained by other frameworks into the MindSpore format, and then convert the model file into a MindSpore model.

<br/>

<font size=3>**Q: Why an error is reported when MindSpore is used to set `model.train`?**</font>

```python
model.train(1, dataset, callbacks=LossMonitor(1), dataset_sink_mode=True)
model.train(1, dataset, callbacks=LossMonitor(1), dataset_sink_mode=False)
```

A: If the offloading mode has been set, it cannot be set to non-offloading mode. This is a restriction on the running mechanism.

<br/>

<font size=3>**Q: What should I pay attention to when using MindSpore to train a model in the `eval` phase? Can the network and parameters be loaded directly? Does the optimizer need to be used in the model?**</font>

A: It mainly depends on what is required in the `eval` phase. For example, the output of the `eval` network of the image classification task is the probability value of each class, and the `acc` is computed with the corresponding label.
In most cases, the training network and parameters can be directly reused. Note that the inference mode needs to be set.

```python
net.set_train(False)
```

The optimizer is not required in the `eval` phase. However, if the `model.eval` API of MindSpore needs to be used, the `loss function` needs to be configured. For example:

```python
# Define a model.
model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
# Evaluate the model.
res = model.eval(dataset)
```

<br/>

<font size=3>**Q: How do I use `param_group` in SGD to reduce the learning rate?**</font>

A: To change the value according to `epoch`, use [Dynamic LR](https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.nn.html#dynamic-lr) and set `step_per_epoch` to `step_size`. To change the value according to `step`, set `step_per_epoch` to 1. You can also use [LearningRateSchedule](https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.nn.html#dynamic-learning-rate).

<br/>

<font size=3>**Q: How do I modify parameters (such as the dropout value) on MindSpore?**</font>

A: When building a network, use `if self.training: x = dropput(x)`. When reasoning, set `network.set_train(mode_false)` before execution to disable the dropout function. During training, set `network.set_train(mode_false)` to True to enable the dropout function.

<br/>

<font size=3>**Q: How do I view the number of model parameters?**</font>

A: You can load the checkpoint to count the parameter number. Variables in the momentum and optimizer may be counted, so you need to filter them out.
You can refer to the following APIs to collect the number of network parameters:

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

[Script Link](https://gitee.com/mindspore/models/blob/master/research/cv/tinynet/src/utils.py).

<br/>

<font size=3>**Q: How do I monitor the loss during training and save the training parameters when the `loss` is the lowest?**</font>

A: You can customize a `callback`.For details, see the writing method of `ModelCheckpoint`. In addition, the logic for determining loss is added.

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

<font size=3>**Q: How do I obtain the expected `feature map` when `nn.Conv2d` is used?**</font>

A: For details about how to derive the `Conv2d shape`, click [here](https://www.mindspore.cn/docs/api/en/r1.6/api_python/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d) Change `pad_mode` of `Conv2d` to `same`. Alternatively, you can calculate the `pad` based on the Conv2d shape derivation formula to keep the `shape` unchanged. Generally, the pad is `(kernel_size-1)//2`.

<br/>

<font size=3>**Q: Can MindSpore be used to customize a loss function that can return multiple values?**</font>

A: After customizing the `loss function`, you need to customize `TrainOneStepCell`. The number of `sens` for implementing gradient calculation is the same as the number of `network` outputs. For details, see the following:

```python
net = Net()
loss_fn = MyLoss()
loss_with_net = MyWithLossCell(net, loss_fn)
train_net = MyTrainOneStepCell(loss_with_net, optim)
model = Model(net=train_net, loss_fn=None, optimizer=None)
```

<br/>

<font size=3>**Q: How does MindSpore implement the early stopping function?**</font>

A: You can customize the `callback` method to implement the early stopping function.
Example: When the loss value decreases to a certain value, the training stops.

```python
class EarlyStop(Callback):
    def __init__(self, control_loss=1):
        super(EarlyStop, self).__init__()
        self._control_loss = control_loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if loss.asnumpy() < self._control_loss:
            # Stop training.
            run_context._stop_requested = True

stop_cb = EarlyStop(control_loss=1)
model.train(epoch_size, ds_train, callbacks=[stop_cb])
```

<br/>

<font size=3>**Q: After a model is trained, how do I save the model output in text or `npy` format?**</font>

A: The network output is `Tensor`. You need to use the `asnumpy()` method to convert the `Tensor` to `NumPy` and then save the data. For details, see the following:

```python
out = net(x)
np.save("output.npy", out.asnumpy())
```

<br/>

<font size=3>**Q: Can the `vgg16` model be loaded and transferred on a GPU using the Hub?**</font>

A: Yes, but you need to manually modify the following two arguments:

```python
# Add the **kwargs argument as follows:
def vgg16(num_classes=1000, args=None, phase="train", **kwargs):
```

```python
# Add the **kwargs argument as follows:
net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase, **kwargs)
```

<br/>

<font size=3>**Q: How to obtain middle-layer features of a VGG model?**</font>

A: Obtaining the middle-layer features of a network is not closely related to the specific framework. For the `vgg` model defined in `torchvison`, the `features` field can be used to obtain the middle-layer features. The `vgg` source code of `torchvison` is as follows:

```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
```

The `vgg16` defined in ModelZoo of MindSpore can be obtained through the `layers` field as follows:

```python
network = vgg16()
print(network.layers)
```

<br/>

<font size=3>**Q: When MindSpore is used for model training, there are four input parameters for `CTCLoss`: `inputs`, `labels_indices`, `labels_values`, and `sequence_length`. How do I use `CTCLoss` for model training?**</font>

A: The `dataset` received by the defined `model.train` API can consist of multiple pieces of data, for example, (`data1`, `data2`, `data3`, ...). Therefore, the `dataset` can contain `inputs`, `labels_indices`, `labels_values`, and `sequence_length` information. You only need to define the dataset in the corresponding format and transfer it to `model.train`. For details, see [Data Processing API](https://www.mindspore.cn/docs/programming_guide/en/r1.6/dataset_loading.html).

<br/>

<font size=3>**Q: How do I load the PyTorch weight to MindSpore during model transfer?**</font>

A: First, enter the `PTH` file of PyTorch. Take `ResNet-18` as an example. The network structure of MindSpore is the same as that of PyTorch. After transferring, the file can be directly loaded to the network. Only `BN` and `Conv2D` are used during loading. If the network names of MindSpore and PyTorch at other layers are different, change the names to the same.

<br/>

<font size=3>**Q: What are the available recommendation or text generation networks or models provided by MindSpore?**</font>

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore ModelZoo](https://gitee.com/mindspore/models/tree/master).

<br/>

<font size=3>**Q: How do I use MindSpore to fit functions such as $f(x)=a \times sin(x)+b$?**</font>

A: The following is based on the official MindSpore linear fitting case.

```python
# The fitting function is: f(x)=2*sin(x)+3.
import numpy as np
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import nn, Model, context
from mindspore.train.callback import LossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def get_data(num, w=2.0, b=3.0):
    # f(x)=w * sin(x) + b
    # f(x)=2 * sin(x) +3
    for i in range(num):
        x = np.random.uniform(-np.pi, np.pi)
        noise = np.random.normal(0, 1)
        y = w * np.sin(x) + b + noise
        yield np.array([np.sin(x)]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":

    num_data = 1600
    batch_size = 16
    repeat_size = 1
    lr = 0.005
    momentum = 0.9

    net = LinearNet()
    net_loss = nn.loss.MSELoss()
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    model = Model(net, net_loss, opt)

    ds_train = create_dataset(num_data, batch_size=batch_size, repeat_size=repeat_size)
    model.train(1, ds_train, callbacks=LossMonitor(), dataset_sink_mode=False)

    print(net.trainable_params()[0], "\n%s" % net.trainable_params()[1])
```

<br/>

<font size=3>**Q: How do I use MindSpore to fit quadratic functions such as $f(x)=ax^2+bx+c$?**</font>

A: The following code is referenced from the official [MindSpore tutorial code](https://gitee.com/mindspore/docs/blob/r1.6/docs/sample_code/linear_regression.py).

Modify the following items to fit $f(x) = ax^2 + bx + c$:

1. Dataset generation.
2. Network fitting.
3. Optimizer.

The following explains detailed information about the modification:

```python
# The selected optimizer does not support CPUs. Therefore, the GPU computing platform is used for training. You need to install MindSpore of the GPU version.
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Assume that the function to be fitted is f(x)=2x^2+3x+4. Modify the data generation function as follows:
def get_data(num, a=2.0, b=3.0 ,c = 4):
    for i in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        # For details about how to generate the value of y, see the to-be-fitted objective function ax^2+bx+c.
        y = x * x * a + x * b + c + noise
        # When fitting a*x^2 + b*x +c, a and b are weight parameters, and c is the offset parameter bias. The training data corresponding to the two weights is x^2 and x, respectively. Therefore, the dataset generation mode is changed as follows:
        yield np.array([x*x, x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        # Two training parameters are input for the full connection function. Therefore, the input value is changed to 2. The first Normal(0.02) automatically allocates random weights to the two input parameters, and the second Normal is the random bias.
        self.fc = nn.Dense(2, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    num_data = 1600
    batch_size = 16
    repeat_size = 1
    lr = 0.005
    momentum = 0.9

    net = LinearNet()
    net_loss = nn.loss.MSELoss()
    # RMSProp optimizer with better effect is selected for quadratic function fitting. Currently, Ascend and GPU computing platforms are supported.
    opt = nn.RMSProp(net.trainable_params(), learning_rate=0.1)
    model = Model(net, net_loss, opt)

    ds_train = create_dataset(num_data, batch_size=batch_size, repeat_size=repeat_size)
    model.train(1, ds_train, callbacks=LossMonitor(), dataset_sink_mode=False)

    print(net.trainable_params()[0], "\n%s" % net.trainable_params()[1])
```

<br/>

<font size=3>**Q: How do I execute a single `ut` case in `mindspore/tests`?**</font>

A: `ut` cases are usually based on the MindSpore package of the debug version, which is not provided on the official website. You can run `sh build.sh` to compile the source code and then run the `pytest` command. The compilation in debug mode does not depend on the backend. Run the `sh build.sh -t on` command. For details about how to execute cases, see the `tests/runtest.sh` script.

<br/>

<font size=3>**Q: For Ascend users, how to get more detailed logs when the `run task error` is reported?**</font>

A: Use the msnpureport tool to set the on-device log level. The tool is stored in `/usr/local/Ascend/driver/tools/msnpureport`.

```bash
- Global: /usr/local/Ascend/driver/tools/msnpureport -g info
```

```bash
- Module-level: /usr/local/Ascend/driver/tools/msnpureport -m SLOG:error
```

```bash
- Event-level: /usr/local/Ascend/driver/tools/msnpureport -e disable/enable
```

```bash
- Multi-device ID-level: /usr/local/Ascend/driver/tools/msnpureport -d 1 -g warning
```

Assume that the value range of deviceID is [0, 7], and `devices 0–3` and `devices 4–7` are on the same OS. `Devices 0–3` share the same log configuration file and `devices 4–7` share the same configuration file. In this way, changing the log level of any device (for example device 0) will change that of other devices (for example `devices 1–3`). This rule also applies to `devices 4–7`.

After the driver package is installed (assuming that the installation path is /usr/local/HiAI and the execution file `msnpureport.exe` is in the C:\ProgramFiles\Huawei\Ascend\Driver\tools\ directory on Windows), run the command in the /home/shihangbo/ directory to export logs on the device to the current directory and store logs in a folder named after the timestamp.

<br/>

<font size=3>**Q: How do I change hyperparameters for calculating loss values during neural network training?**</font>

A: Sorry, this function is not available yet. You can find the optimal hyperparameters by training, redefining an optimizer, and then training.

<br/>

<font size=3>**Q: What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A: While installing Ascend 310 AI Processor software packages，the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

<br/>

<font size=3>**Q: Why does context.set_ps_context(enable_ps=True) in model_zoo/official/cv/resnet/train.py in the MindSpore code have to be set before init?**</font>

A: In MindSpore Ascend mode, if init is called first, then all processes will be allocated cards, but in parameter server training mode, the server does not need to allocate cards, then the worker and server will use the same card, resulting in an error: HCCL dependent tsd is not open.

<br/>

<font size=3>**Q: What should I do if the memory continues to increase when resnet50 training is being performed on the CPU ARM platform?**</font>

A: When performing resnet50 training on the CPU ARM, some operators are implemented based on the oneDNN library, and the oneDNN library is based on the libgomp library to achieve multi-threaded parallelism. Currently, libgomp has multiple parallel domain configurations. The number of threads is different and the memory usage continues to increase. The continuous growth of memory can be controlled by configuring a uniform number of threads globally. For comprehensive performance considerations, it is recommended to configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`.

<font size=3>**Q: Why report an error that the stream exceeds the limit when executing the model on the Ascend platform？**</font>

A: Stream represents an operation queue. Tasks on the same stream are executed in sequence, and different streams can be executed in parallel. Various operations in the network generate tasks and are assigned to streams to control the concurrent mode of task execution. Ascend platform has a limit on the number of tasks on the same stream, and tasks that exceed the limit will be assigned to new streams. The multiple parallel methods of MindSpore will also assign new streams, such as parallel communication operators. Therefore, when the number of assigned streams exceeds the resource limit of the Ascend platform, an error will be reported. Reference solution:

- Reduce the size of the network model

- Reduce the use of communication operators in the network

- Reduce conditional control statements in the network

<br/>

<font size=3>**Q: On the Ascend platform, if an error "Ascend error occurred, error message:" is reported and followed by an error code, such as "E40011", how to find the cause of the error code?**</font>

A: When "Ascend error occurred, error message:" appears, it indicates that a module of Ascend CANN is abnormal and the error code is reported.

At this time, there is an error message after the error code. If you need a more detailed possible cause and solution for this exception, please refer to the "error code troubleshooting" section of the corresponding Ascend version document, such as [CANN Community 5.0.3 alpha 002 (training) Error Code troubleshooting](https://support.huaweicloud.com/trouble-cann503alpha2training/atlaspd_15_0001.html).

<br/>

<font size=3>**Q: When the third-party component gensim is used to train the NLP network, the error "ValueError" may be reported. What can I do?**</font>

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

<font size=3>**Q: What should I do if I encounter `matplotlib.pyplot.show()` (most often plt.show()) cannot be executed during the tutorial is running?**</font>

A: First confirm whether `matplotlib` is installed. If it is not installed, you can execute `pip install matplotlib` on the command line to install it.

Secondly, because the function of `matplotlib.pyplot.show()` is to display graph data graphically, it is necessary to run the system to support the graph display function. If the system cannot support graph display, the reader needs to comment out the command line of the graph display. Operation will not affect the results of the overall code.

<br/>

<font size=3>**Q: What issues should be paid attention to when using the *Run in ModelArts* in tutorials?**</font>

A: Need to confirm that the following preparations have been done.

- First, you need to log in to ModelArts through your HUAWEI CLOUD account.
- Secondly, note that the hardware environment supported by the tags in the tutorial document is Ascend, GPU or CPU. Since the hardware environment used by default after login is CPU, the Ascend environment and GPU environment need to be switched manually by the user.
- Finally, confirm that the current `Kernel` of Jupyter Notebook is MindSpore.

After completing the above steps, you can run the tutorial.

For the specific operation process, please refer to [Based on ModelArts Online Experience MindSpore](https://bbs.huaweicloud.com/forum/thread-168982-1-1.html).

<br/>

<font size=3>**Q: No error is reported when using result of division in GRAPH mode, but an error is reported when using result of division in PYNATIVE mode？**</font>

A: In GRAPH mode, the data type of the output result of the operator is determined at the graph compilation stage.

For example, the following code is executed in GRAPH mode, the type of input data is int type, so the output result is also int type according to graph compiler.

```python
from mindspore import context
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

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

output：

```text
4 <class 'int'>
```

Change GRAPH_MODE to PYNATIVE_MODE. Since the Python syntax is used in PyNative mode, the type of any division output is float type, so the execution result is as follows.

```text
4.0 <class 'float'>
```

Therefore, in the scenario where the subsequent operator clearly needs to use int, it is recommended to use Python's divisibility symbol `//`.

<br/>
