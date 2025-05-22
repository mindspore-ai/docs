# Implementing the Model Fault Injection and Evaluation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/fault_injection.md)

## Overview

In the past decade, artificial intelligence has become ubiquitous in many applications.
It is also being increasingly deployed in safety critical or security critical applications
such as automatic driving, intelligent security, intelligent medical treatment and so on.
In these domains, it is critical to ensure the reliability of the AI models and its
implementation as faults can lead to loss of life and property.

In order to ensure the reliability and availability of AI model under various fault scenarios,
it is important to strictly test and verify its components.
This module can simulate various fault scenarios and evaluation of model reliability.

The following is a simple example showing the overall process of model fault injection and evaluation:

1. Download a public dataset.
2. Prepare both datasets and pre-train models.
3. Call the fault injection module.
4. View the execution result.

> You can obtain the complete executable code at <https://gitee.com/mindspore/mindarmour/blob/master/examples/reliability/model_fault_injection.py>

## Preparations

Ensure that the MindSpore is correctly installed. If not, install MindSpore by following the [Installation Guide](https://www.mindspore.cn/install/en).

### Downloading the Dataset

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
> Download the dataset at <http://yann.lecun.com/exdb/mnist/>

Decompress the downloaded dataset to a local path. The directory structure is as follows:

```text
- data_path
    - train
        - train-images-idx3-ubyte
        - train-labels-idx1-ubyte
    - test
        - t10k-images-idx3-ubyte
        - t10k-labels-idx1-ubyte
```

### Downloading the Checkpoint File

Download checkpoint file or just trained your own checkpoint.
> Download the checkpoint file at <https://www.mindspore.cn/hub>

### Importing the Python Library and Modules

Before start, you need to import the Python library.

```python
import numpy as np
import mindspore as ms

from mindspore.train import Model
from mindarmour.reliability import FaultInjector
from examples.common.networks.lenet5.lenet5_net import LeNet5
from examples.common.dataset.data_processing import generate_mnist_dataset
```

## Constructing the Dataset and Model

Take MNIST dataset and LeNet5 as an example.

Construct MNIST dataset:

```python
DATA_FILE = 'PATH_TO_MNIST/'
ds_eval = generate_mnist_dataset(DATA_FILE, batch_size=64)
test_images = []
test_labels = []
for data in ds_eval.create_tuple_iterator(output_numpy=True):
    images = data[0].astype(np.float32)
    labels = data[1]
    test_images.append(images)
    test_labels.append(labels)
ds_data = np.concatenate(test_images, axis=0)
ds_label = np.concatenate(test_labels, axis=0)
```

Construct LeNet5:

```python
ckpt_path = 'PATH_TO_CHECKPOINT/'
net = LeNet5()
param_dict = ms.load_checkpoint(ckpt_path)
ms.load_param_into_net(net, param_dict)
model = Model(net)
```

## Setup Parameters and Initialize Fault Injection Module

Setup parameters, the code is as follows:

```python
fi_type = ['bitflips_designated', 'precision_loss']
fi_mode = ['single_layer', 'all_layer']
fi_size = [1, 2]
```

Initialize fault injection module:

```python
fi = FaultInjector(model=model, fi_type=fi_type, fi_mode=fi_mode, fi_size=fi_size)
```

The initialization parameters are described as follows:

- `model(Model)`: The model needs to be evaluated.
- `fi_type(list)`: The type of the fault injection which includes `bitflips_random`(flip randomly),
            `bitflips_designated`(flip the key bit), `random`, `zeros`, `NaN`, `INF`, `anti_activation` `precision_loss` etc.
    - `bitflips_random`: Bits are flipped randomly in the chosen value.
    - `bitflips_designated`: Specified bit is flipped in the chosen value.
    - `random`: The chosen value are replaced with random value in the range [-1, 1].
    - `zeros`: The chosen value are replaced with zero.
    - `NaN`: The chosen value are replaced with NaN.
    - `INF`: The chosen value are replaced with INF.
    - `anti_activation`: Changing the sign of the chosen value.
    - `precision_loss`: Round the chosen value to 1 decimal place.
- `fi_mode(list)`: There are twe kinds of injection modes can be specified, `single_layer` or `all_layer`.
- `fi_size(list)`: The exact number of values to be injected with the specified fault. For `zeros`, `anti_activation` and `precision_loss` fault, `fi_size` is the percentage of total tensor values and varies from 0% to 100%.

## Evaluation

After the module is initialized, call the fault injection function `kick_off`.

```python
results = fi.kick_off(ds_data, ds_label, iter_times=100)
```

- `ds_data(numpy.ndarray)`: The data for testing. The fault tolerance of the model will be evaluated on this data.
- `ds_label(numpy.ndarray)`: The label of data, corresponding to the data.
- `iter_times(numpy.ndarray)`: The number of evaluations, which will determine the batch size.

call function `metrics`, and get summary result:

```python
result_summary = fi.metrics()
```

Return:

- `results(list)`: The Evaluation results of each parameter.
- `result_summary(list)`: Summary results are counted according to the fi_mode.

## Viewing the Result

```python
for result in results:
    print(result)
for result in result_summary:
    print(result)
```

The result is as follows:

```text
{'original_acc': 0.9797676282051282}
{'type': 'bitflips_designated', 'mode': 'single_layer', 'size': 1, 'acc': 0.7028245192307693, 'SDC': 0.2769431089743589}
{'type': 'bitflips_designated', 'mode': 'single_layer', 'size': 2, 'acc': 0.5052083333333334, 'SDC': 0.4745592948717948}
{'type': 'bitflips_designated', 'mode': 'all_layer', 'size': 1, 'acc': 0.2077323717948718, 'SDC': 0.7720352564102564}
{'type': 'bitflips_designated', 'mode': 'all_layer', 'size': 2, 'acc': 0.15745192307692307, 'SDC': 0.8223157051282051}
{'type': 'precision_loss', 'mode': 'single_layer', 'size': 1, 'acc': 0.9795673076923077, 'SDC': 0.00020032051282048435}
{'type': 'precision_loss', 'mode': 'single_layer', 'size': 2, 'acc': 0.9797676282051282, 'SDC': 0.0}
{'type': 'precision_loss', 'mode': 'all_layer', 'size': 1, 'acc': 0.9794671474358975, 'SDC': 0.00030048076923072653}
{'type': 'precision_loss', 'mode': 'all_layer', 'size': 2, 'acc': 0.9795673076923077, 'SDC': 0.00020032051282048435}
single_layer_acc_mean:0.791842 single_layer_acc_max:0.979768 single_layer_acc_min:0.505208
single_layer_SDC_mean:0.187926 single_layer_SDC_max:0.474559 single_layer_SDC_min:0.000000
all_layer_acc_mean:0.581055 all_layer_acc_max:0.979567 all_layer_acc_min:0.157452
all_layer_SDC_mean:0.398713 all_layer_SDC_max:0.822316 all_layer_SDC_min:0.000200
```

- `original_acc`: The original accuracy of model.
- `SDC(Silent Data Corruption)`: The difference between the original accuracy and the current fault accuracy.
- `single_layer_acc_mean/max/min`: The average/maximum/minimum accuracy in single_layer mode.
- `single_layer_SDC_mean/max/min`: The average/maximum/minimum SDC in single_layer mode.
- `all_layer_acc_mean/max/min`: The average/maximum/minimum accuracy in all_layer mode.
- `all_layer_SDC_mean/max/min`: The average/maximum/minimum SDC in all_layer mode.