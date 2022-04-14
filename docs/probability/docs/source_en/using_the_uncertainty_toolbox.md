# Using the Uncertainty Evaluation Toolbox

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/probability/docs/source_en/using_the_uncertainty_toolbox.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

One of advantages of BNN is that uncertainty can be obtained. MDP provides a toolbox for uncertainty evaluation at the upper layer. Users can easily use the toolbox to compute uncertainty. Uncertainty means an uncertain degree of a prediction result of a deep learning model. Currently, most deep learning algorithm can only provide prediction results but cannot determine the result reliability. There are two types of uncertainties: aleatoric uncertainty and epistemic uncertainty.

- Aleatoric uncertainty: Internal noises in data, that is, unavoidable errors. This uncertainty cannot be reduced by adding sampling data.
- Epistemic uncertainty: An inaccurate evaluation of input data by a model due to reasons such as poor training or insufficient training data. This may be reduced by adding training data.

The uncertainty evaluation toolbox is applicable to mainstream deep learning models, such as regression and classification. During inference, developers can use the toolbox to obtain any aleatoric uncertainty and epistemic uncertainty by training models and training datasets and specifying tasks and samples to be evaluated. Developers can understand models and datasets based on uncertainty information.

This example will use the MNIST dataset and the LeNet5 network model example for this experience.

1. Data preparation.
2. Define a deep learning network.
3. Initialize the uncertainty assessment toolbox.
4. Assess cognitive uncertainty.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/r1.7/tests/st/probability/toolbox>.

## Data Preparation

### Downloading the Dataset

Download the MNIST_Data dataset, execute the following command to complete the download of the dataset and unzip it to the specified location:

```python
import os
import requests

def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)
```

```text
./datasets/MNIST_Data
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte

2 directories, 4 files
```

### Data Enhancement

Define the dataset enhancement function, and enhance the original data into data suitable for the LeNet network.

```python
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define some parameters needed for data enhancement and rough justification
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # according to the parameters, generate the corresponding data enhancement method
    c_trans = [
        CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR),
        CV.Rescale(rescale_nml, shift_nml),
        CV.Rescale(rescale, shift),
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(mstype.int32)

    # using map to apply operations to a dataset
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=num_parallel_workers)

    # process the generated dataset
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds
```

## Defining a Deep Learning Network

This example uses the LeNet5 deep neural network, which is implemented in MindSpore as follows:

```python
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    """Lenet network structure."""
    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Initializing the Uncertainty Toolbox

Initialize the `UncertaintyEvaluation` function of the Uncertainty Toolbox and prepare as follows:

1. Prepare the model weight parameter file.
2. Load the model weight parameter file into the neural network.
3. Enhance the training dataset into data suitable for neural networks.
4. Load the above network and dataset into UncertaintyEvaluation.

MindSpore uses the Uncertainty Toolbox `UncertaintyEvaluation` interface to measure model accidental uncertainty and cognitive uncertainty. For more usage methods, please refer to [API](https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore.nn.probability.html#module-mindspore.nn.probability.toolbox).

### Preparing the Model Weight Parameter File

In this example, the corresponding model weight parameter file `checkpoint_lenet.ckpt` has been prepared. This parameter file is the weight parameter file saved after training for 5 epochs in [Quick Start for Beginners](https://www.mindspore.cn/tutorials/en/r1.7/beginner/quick_start.html), execute the following command to download:

```bash
download_dataset("https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/models/checkpoint_lenet.ckpt", ".")
```

### Completing the Initialization

Load the DNN network and the training dataset that need uncertainty measurement. Since the uncertainty measurement requires a Bayesian network, when the initialized uncertainty measurement tool is called for the first time, the DNN network will be converted to The Bayesian network is trained, and after completion, the corresponding data can be passed in to measure accidental uncertainty or cognitive uncertainty.

```python
from mindspore import context
from mindspore.nn.probability.toolbox import UncertaintyEvaluation
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# get trained model
network = LeNet5()
param_dict = load_checkpoint('checkpoint_lenet.ckpt')
load_param_into_net(network, param_dict)
# get train
ds_train = create_dataset('./datasets/MNIST_Data/train')
evaluation = UncertaintyEvaluation(model=network,
                                   train_dataset=ds_train,
                                   task_type='classification',
                                   num_classes=10,
                                   epochs=1,
                                   epi_uncer_model_path=None,
                                   ale_uncer_model_path=None,
                                   save_model=False)
```

## Cognitive Uncertainty Assessment

### Converting to Bayesian Training Measurement

First, take out a `batch` of the verification dataset to measure the cognitive uncertainty, and convert the original deep neural network into a Bayesian network for training when it is first called.

```python
ds_test = create_dataset("./datasets/MNIST_Data/test")
batch_data = next(ds_test.create_dict_iterator())
eval_images = batch_data["image"]
eval_labels = batch_data["label"]
epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_images)
```

```text
epoch: 1 step: 1, loss is 0.14702837
epoch: 1 step: 2, loss is 0.00017862688
epoch: 1 step: 3, loss is 0.09421586
epoch: 1 step: 4, loss is 0.0003434865
epoch: 1 step: 5, loss is 7.1358285e-05
... ...
epoch: 1 step: 1871, loss is 0.20069705
epoch: 1 step: 1872, loss is 0.12135945
epoch: 1 step: 1873, loss is 0.04572148
epoch: 1 step: 1874, loss is 0.04962858
epoch: 1 step: 1875, loss is 0.0019944885
```

`evaluation.eval_epistemic_uncertainty`: Cognitive uncertainty measurement interface. The training data will be used to convert the DNN model into Bayesian training when it is first called.

`eval_images`: The `batch` images used in the occasional uncertainty test.

### Printing the Cognitive Uncertainty

Take a batch of data and print out the label.

```python
print(eval_labels)
print(epistemic_uncertainty.shape)
```

```text
[2 9 4 3 9 9 2 4 9 6 0 5 6 8 7 6 1 9 7 6 5 4 0 3 7 7 6 7 7 4 6 2]
(32, 10)
```

The cognitive uncertainty content is the uncertainty value of the classification model from 0-9 corresponding to 32 pictures.

Take the first two pictures and print out the accidental uncertainty value of the corresponding model.

```python
print("the picture one, number is {}, epistemic uncertainty is:\n{}".format(eval_labels[0], epistemic_uncertainty[0]))
print("the picture two, number is {}, epistemic uncertainty is:\n{}".format(eval_labels[1], epistemic_uncertainty[1]))
```

```text
the picture one, number is 2, epistemic uncertainty is:
[0.75372726 0.2053496  3.737096   0.7113453  0.93452704 0.40339947
 0.91918266 0.44237098 0.40863538 0.8195221 ]
the picture two, number is 9, epistemic uncertainty is:
[0.97602427 0.37808532 0.4955423  0.17907992 1.3365419  0.20227651
 2.2211757  0.27501273 0.30733848 3.7536747 ]
 ```
