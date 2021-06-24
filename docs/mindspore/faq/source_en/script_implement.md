# Script Implement

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/faq/source_en/script_implement.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

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

A: The `dataset` received by the defined `model.train` API can consist of multiple pieces of data, for example, (`data1`, `data2`, `data3`, ...). Therefore, the `dataset` can contain `inputs`, `labels_indices`, `labels_values`, and `sequence_length` information. You only need to define the dataset in the corresponding format and transfer it to `model.train`. For details, see [Data Processing API](https://www.mindspore.cn/docs/programming_guide/en/r1.3/dataset_loading.html).

<br/>

<font size=3>**Q: How do I load the PyTorch weight to MindSpore during model transfer?**</font>

A: First, enter the `PTH` file of PyTorch. Take `ResNet-18` as an example. The network structure of MindSpore is the same as that of PyTorch. After transferring, the file can be directly loaded to the network. Only `BN` and `Conv2D` are used during loading. If the network names of MindSpore and PyTorch at other layers are different, change the names to the same.

<br/>

<font size=3>**Q: What are the available recommendation or text generation networks or models provided by MindSpore?**</font>

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

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

A: The following code is referenced from the official [MindSpore tutorial code](https://gitee.com/mindspore/docs/blob/master/tutorials/tutorial_code/linear_regression.py).

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

<font size=3>**Q: What can I do if the Google's Chrome browser prompts the error message `ERR_UNSAFE_PORT after` MindInsight is successfully started?**</font>

A: Chrome browser's kernel prohibits certain ports from being used as HTTP services. You can add `--explicitly-allowed-ports=port` in Chrome browser's configuration. Otherwise you can change the port or browser like IE browser.

<br/>

<font size=3>**Q: How do I change hyperparameters for calculating loss values during neural network training?**</font>

A: Sorry, this function is not available yet. You can find the optimal hyperparameters by training, redefining an optimizer, and then training.

<br/>

<font size=3>**Q：What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A：While installing Ascend 310 AI Processor software packages，the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

<br/>

<font size=3>**Q：Why does context.set_ps_context(enable_ps=True) in model_zoo/official/cv/resnet/train.py in the MindSpore code have to be set before init?**</font>

A：In MindSpore Ascend mode, if init is called first, then all processes will be allocated cards, but in parameter server training mode, the server does not need to allocate cards, then the worker and server will use the same card, resulting in an error: Hccl dependent tsd is not open.