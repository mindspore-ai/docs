[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/quick_start.md)

[Introduction](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/introduction.html) || **Quick Start** || [Tensor](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/train.html) || [Save and load](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/mixed_precision.html)

# Quick Start

This section quickly implements a simple deep learning model through MindSpore APIs. For a deeper understanding of how to use MindSpore, see the reference links provided at the end of each section.

```python
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
```

## Processing a Dataset

MindSpore provides Pipeline-based [Data Engine](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/design/data_engine.html) and achieves efficient data preprocessing through [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html). In this tutorial, we use the Mnist dataset and pre-process dataset by using the data transformations provided by `mindspore.dataset`, after automatically downloaded.

> The sample code in this chapter relies on `download`, which can be installed by using the command `pip install download`. If this document is run as Notebook, you need to restart the kernel after installation to execute subsequent code.

```python
# Download data from open datasets\n",
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:01<00:00, 6.73MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

The directory structure of MNIST Dataset is as following:

```text
MNIST_Data
└── train
    ├── train-images-idx3-ubyte (60000 training images)
    ├── train-labels-idx1-ubyte (60000 training labels)
└── test
    ├── t10k-images-idx3-ubyte (10000 test images)
    ├── t10k-labels-idx1-ubyte (10000 test labels)
```

After the data is downloaded, the dataset object is obtained.

```python
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
```

Print the names of the data columns contained in the dataset for dataset pre-processing.

```python
print(train_dataset.get_col_names())
```

```text
['image', 'label']
```

Dataset in MindSpore uses the Data Processing Pipeline, which requires specifying operations such as map, batch, and shuffle. Here we use map to transform the image data and labels by scaling the input image to 1/255 and normalizing it according to the mean value of 0.1307 and the standard deviation value of 0.3081, and then the processed dataset is packed into a batch of size 64.

```python
def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]

    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset
```

```python
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)
```

[create_tuple_iterator](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_tuple_iterator.html) or [create_dict_iterator](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_dict_iterator.html) could be used to iterate over the dataset, printing the shape and dtype for `image` and `label`.

```python
for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    break
```

```text
Shape of image [N, C, H, W]: (64, 1, 28, 28) Float32
Shape of label: (64,) Int32
```

```python
for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break
```

```text
Shape of image [N, C, H, W]: (64, 1, 28, 28) Float32
Shape of label: (64,) Int32
```

For more detailed information, see [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html).

## Building Network

`mindspore.nn` class is the base class for building all networks and is the basic unit of the network. When the user needs to customize the network, you can inherit the `nn.Cell` class and override the `__init__` method and the `construct` method. `__init__` contains the definitions of all network layers, and `construct` contains the transformation process of the data ([Tensor](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/tensor.html)).

```python
# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
print(model)
```

```text
Network<
  (flatten): Flatten<>
  (dense_relu_sequential): SequentialCell<
    (0): Dense<input_channels=784, output_channels=512, has_bias=True>
    (1): ReLU<>
    (2): Dense<input_channels=512, output_channels=512, has_bias=True>
    (3): ReLU<>
    (4): Dense<input_channels=512, output_channels=10, has_bias=True>
    >
  >
```

For more detailed information, see [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html).

## Training Model

In model training, a complete training process (step) requires the following three steps:

1. **Forward calculation**: model predicts results (logits) and finds the prediction loss (loss) with the correct label (label).
2. **Backpropagation**: Using an automatic differentiation mechanism, the gradients of the model parameters (parameters) with respect to the loss are automatically found.
3. **Parameter optimization**: update the gradient to the parameter.

MindSpore uses a functional automatic differentiation mechanism, implemented through the steps above:

1. Define forward calculation function.
2. Obtain the gradient calculation function by function transformation, calling [value_and_grad](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.value_and_grad.html) for details.
3. Define training functions, set to training mode by calling [set_train](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.set_train) for setting of training mode, and perform forward computation, back propagation and parameter optimization.

```python
# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

# 1. Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# 2. Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 3. Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
```

In addition to training, we define test functions that are used to evaluate the performance of the model.

```python
def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process requires several iterations of the dataset, and one complete iteration is called an epoch. In each round, the training set is traversed for training and the test set is used for prediction at the end. The loss value and prediction accuracy (Accuracy) of each round are printed, and it can be seen that the loss is decreasing and Accuracy is increasing.

```python
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)
    test(model, test_dataset, loss_fn)
print("Done!")
```

```text
Epoch 1
-------------------------------
loss: 2.302088  [  0/938]
loss: 2.290692  [100/938]
loss: 2.266338  [200/938]
loss: 2.205240  [300/938]
loss: 1.907198  [400/938]
loss: 1.455603  [500/938]
loss: 0.861103  [600/938]
loss: 0.767219  [700/938]
loss: 0.422253  [800/938]
loss: 0.513922  [900/938]
Test:
 Accuracy: 83.8%, Avg loss: 0.529534

Epoch 2
-------------------------------
loss: 0.580867  [  0/938]
loss: 0.479347  [100/938]
loss: 0.677991  [200/938]
loss: 0.550141  [300/938]
loss: 0.226565  [400/938]
loss: 0.314738  [500/938]
loss: 0.298739  [600/938]
loss: 0.459540  [700/938]
loss: 0.332978  [800/938]
loss: 0.406709  [900/938]
Test:
 Accuracy: 90.2%, Avg loss: 0.334828

Epoch 3
-------------------------------
loss: 0.461890  [  0/938]
loss: 0.242303  [100/938]
loss: 0.281414  [200/938]
loss: 0.207835  [300/938]
loss: 0.206000  [400/938]
loss: 0.409646  [500/938]
loss: 0.193608  [600/938]
loss: 0.217575  [700/938]
loss: 0.212817  [800/938]
loss: 0.202862  [900/938]
Test:
 Accuracy: 91.9%, Avg loss: 0.280962

Done!
```

For the detailed information, see [Train](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/train.html).

## Saving a Model

After the model is trained, its parameters need to be saved.

```python
# Save checkpoint
mindspore.save_checkpoint(model, "model.ckpt")
print("Saved Model to model.ckpt")
```

```text
Saved Model to model.ckpt
```

## Loading a Model

There are two steps to load the saved weights:

1. Reinstantiate the model object and construct the model.
2. Load the model parameters and load them onto the model.

```python
# Instantiate a random initialized model
model = Network()
# Load checkpoint and load parameter to model
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)
```

```text
[]
```

> `param_not_load` is an unloaded parameter list. When the list is empty, it means all parameters are loaded successfully.

The loaded model can be used directly for predictive inference.

```python
model.set_train(False)
for data, label in test_dataset:
    pred = model(data)
    predicted = pred.argmax(1)
    print(f'Predicted: "{predicted[:10]}", Actual: "{label[:10]}"')
    break
```

```text
Predicted: "[3 9 6 1 6 7 4 5 2 2]", Actual: "[3 9 6 1 6 7 4 5 2 2]"
```

For more detailed information, see [Save and Load](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html).
