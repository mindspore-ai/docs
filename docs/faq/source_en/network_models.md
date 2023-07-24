# Network Models

`Data Processing` `Environmental Setup` `Model Export` `Model Training` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_en/network_models.md)

<font size=3>**Q: How do I understand the `dataset_sink_mode` parameter in `model.train` of MindSpore?**</font>

A: When `dataset_sink_mode` is set to `True`, data processing and network computing are performed in pipeline mode. That is, when data processing is performed step by step, after a `batch` of data is processed, the data is placed in a queue which is used to cache the processed data. Then, network computing obtains data from the queue for training. In this case, data processing and network computing are performed in pipeline mode. The entire training duration is the longest data processing/network computing duration.

When `dataset_sink_mode` is set to `False`, data processing and network computing are performed in serial mode. That is, after a `batch` of data is processed, it is transferred to the network for computation. After the computation is complete, the next `batch` of data is processed and transferred to the network for computation. This process repeats until the training is complete. The total time consumed is the time consumed for data processing plus the time consumed for network computing.

<br/>

<font size=3>**Q: Can MindSpore train image data of different sizes by batch?**</font>

A: You can refer to the usage of YOLOv3 which contains the resizing of different images. For details about the script, see [yolo_dataset](https://gitee.com/mindspore/mindspore/blob/r1.2/model_zoo/official/cv/yolov3_darknet53/src/yolo_dataset.py).

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

A: The `dataset` received by the defined `model.train` API can consist of multiple pieces of data, for example, (`data1`, `data2`, `data3`, ...). Therefore, the `dataset` can contain `inputs`, `labels_indices`, `labels_values`, and `sequence_length` information. You only need to define the dataset in the corresponding format and transfer it to `model.train`. For details, see [Data Processing API](https://www.mindspore.cn/doc/programming_guide/en/r1.2/dataset_loading.html).

<br/>

<font size=3>**Q: How do I load the PyTorch weight to MindSpore during model transfer?**</font>

A: First, enter the `PTH` file of PyTorch. Take `ResNet-18` as an example. The network structure of MindSpore is the same as that of PyTorch. After transferring, the file can be directly loaded to the network. Only `BN` and `Conv2D` are used during loading. If the network names of MindSpore and PyTorch at other layers are different, change the names to the same.

<br/>

<font size=3>**Q: After a model is trained, how do I save the model output in text or `npy` format?**</font>

A: The network output is `Tensor`. You need to use the `asnumpy()` method to convert the `Tensor` to `NumPy` and then save the data. For details, see the following:

```python
out = net(x)

np.save("output.npy", out.asnumpy())
```

<br/>

<font size=3>**Q: Must data be converted into MindRecords when MindSpore is used for segmentation training?**</font>

A: [build_seg_data.py](https://github.com/mindspore-ai/mindspore/blob/r1.2/model_zoo/official/cv/deeplabv3/src/data/build_seg_data.py) is used to generate MindRecords based on a dataset. You can directly use or adapt it to your dataset. Alternatively, you can use `GeneratorDataset` if you want to read the dataset by yourself.

[GenratorDataset example](https://www.mindspore.cn/doc/programming_guide/en/r1.2/dataset_loading.html#loading-user-defined-dataset)

[GeneratorDataset API description](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

<br/>

<font size=3>**Q: Can MindSpore read a TensorFlow checkpoint?**</font>

A: The checkpoint format of MindSpore is different from that of TensorFlow. Although both use the Protocol Buffers, their definitions are different. Currently, MindSpore cannot read the TensorFlow or Pytorch checkpoints.

<br/>

<font size=3>**Q: How do I perform training without processing data in MindRecord format?**</font>

A: You can use the customized data loading method `GeneratorDataset`. For details, click [here](https://www.mindspore.cn/tutorial/en/r0.7/use/data_preparation/loading_the_datasets.html#id5).

<br/>

<font size=3>**Q: What framework models and formats can be directly read by MindSpore? Can the PTH Model Obtained Through Training in PyTorch Be Loaded to the MindSpore Framework for Use?**</font>

A: MindSpore uses protocol buffers (protobuf) to store training parameters and cannot directly read framework models. A model file stores parameters and their values. You can use APIs of other frameworks to read parameters, obtain the key-value pairs of parameters, and load the key-value pairs to MindSpore. If you want to use the .ckpt file trained by a framework, read the parameters and then call the `save_checkpoint` API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

<font size=3>**Q: How do I use models trained by MindSpore on Ascend 310? Can they be converted to models used by HiLens Kit?**</font>

A: Yes. HiLens Kit uses Ascend 310 as the inference core. Therefore, the two questions are essentially the same. Ascend 310 requires a dedicated OM model. Use MindSpore to export the ONNX or AIR model and convert it into an OM model supported by Ascend 310. For details, see [Multi-platform Inference](https://www.mindspore.cn/tutorial/inference/en/r1.2/multi_platform_inference_ascend_310.html).

<br/>

<font size=3>**Q: How do I modify parameters (such as the dropout value) on MindSpore?**</font>

A: When building a network, use `if self.training: x = dropput(x)`. During verification, set `network.set_train(mode_false)` before execution to disable the dropout function. During training, set `network.set_train(mode_false)` to True to enable the dropout function.

<br/>

<font size=3>**Q: Where can I view the sample code or tutorial of MindSpore training and inference?**</font>

A: Please visit the [MindSpore official website training](https://www.mindspore.cn/tutorial/training/en/r1.2/index.html) and [MindSpore official website inference](https://www.mindspore.cn/tutorial/inference/en/r1.2/index.html).

<br/>

<font size=3>**Q: What types of model is currently supported by MindSpore for training?**</font>

A: MindSpore has basic support for common training scenarios, please refer to [Release note](https://gitee.com/mindspore/mindspore/blob/r1.2/RELEASE.md#) for detailed information.

<br/>

<font size=3>**Q: What are the available recommendation or text generation networks or models provided by MindSpore?**</font>

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo).

<br/>

<font size=3>**Q: How simple can the MindSpore model training code be?**</font>

A: MindSpore provides Model APIs except for network definitions. In most scenarios, model training can be completed using only a few lines of code.

<br/>

<font size=3>**Q: How do I use MindSpore to fit functions such as $f(x)=a \times sin(x)+b$?**</font>

A: The following is based on the official MindSpore linear fitting case.

```python
# The fitting function is：f(x)=2*sin(x)+3.
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

A: The following code is referenced from the official [MindSpore tutorial code](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/tutorial_code/linear_regression.py).

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

<font size=3>**Q：What should I do if a Protobuf memory limit error is reported during the process of using ckpt or exporting a model?**</font>

A：When a single Protobuf data is too large, because Protobuf itself limits the size of the data stream, a memory limit error will be reported. At this time, the restriction can be lifted by setting the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

<font size=3>**Q: When the third-party component gensim is used to train the NLP network, the error "ValueError" may be reported. What can I do? **</font>

A：The following error information is displayed:

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