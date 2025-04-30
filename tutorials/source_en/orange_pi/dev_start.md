# Quick Start

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/orange_pi/dev_start.md)

Since developers may perform custom model and case development in OrangePi AIpro (hereinafter: OrangePi Development Board), this chapter illustrates the development considerations in the OrangePi Development Board through a handwritten digit recognition case based on MindSpore.

## Preparing Running Environment

After obtaining the OrangePi AIpro development board, developers first need to confirm hardware resources, burn images, and upgrade CANN and MindSpore versions before running the case. The specific steps are as follows:

| OrangePi AIpro | Image | CANN Toolkit/Kernels | MindSpore |
| :----:| :----: | :----:| :----: |
| 8T 16G | Ubuntu | 8.0.RC3.alpha002| 2.4.10 |
| 8T 16G | Ubuntu | 8.0.0beta1| 2.5.0 |

### Image Burning

To run this case, it is necessary to burn the Ubuntu image on the OrangePi AIpro official website. Please refer to [Image Burning](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/orange_pi/environment_setup.html#1-image-burning-taking-windows-as-an-example).

### CANN Upgrading

Please refer to [CANN Upgrading](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/orange_pi/environment_setup.html#3-cann-upgrading).

### MindSpore Upgrading

Please refer to [MindSpore Upgrading](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/orange_pi/environment_setup.html#4-mindspore-upgrading).

```python
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
```

## Setting Running Environment

Due to resource constraints, performance optimization mode needs to be enabled with the following parameters:

max_device_memory="2GB": Set the maximum memory available to the device to 2GB.

mode=mindspore.GRAPH_MODE: Indicates running in GRAPH_MODE mode.

device_target="Ascend": Indicates that the target device to be run is Ascend.

jit_config={"jit_level":"O2"}: The compilation optimization level turns on extreme performance optimization and uses sinking execution.

ascend_config={"precision_mode":"allow_mix_precision"}: Auto mixed-precision, which automatically reduces the precision of some operators to float16 or bfloat16.

```python
import mindspore
mindspore.set_context(max_device_memory="2GB", mode=mindspore.GRAPH_MODE, device_target="Ascend",  jit_config={"jit_level":"O2"}, ascend_config={"precision_mode":"allow_mix_precision"})
```

## Preparing and Loading Dataset

MindSpore provides a Pipeline-based [data engine](https://www.mindspore.cn/docs/en/r2.6.0rc1/design/data_engine.html) to realize efficient data preprocessing through [data loading and processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) to realize efficient data preprocessing. In this case, we use the Mnist dataset, which is automatically downloaded and then preprocessed using the data transforms provided by `mindspore.dataset`.

```python
#install download

!pip install download
```

```python
# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:01<00:00, 7.63MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

The MNIST dataset catalog is structured as follows:

```text
MNIST_Data
└── train
    ├── train-images-idx3-ubyte (60000 training images)
    ├── train-labels-idx1-ubyte (60000 training labels)
└── test
    ├── t10k-images-idx3-ubyte (10000 test images)
    ├── t10k-labels-idx1-ubyte (10000 test labels)
```

After the data download is complete, the dataset object is obtained.

```python
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
```

Prints the names of the data columns contained in the dataset for dataset preprocessing.

```python
print(train_dataset.get_col_names())
```

```text
['image', 'label']
```

MindSpore dataset using data processing pipeline needs to specify map, batch, shuffle and other operations. Here we use map to transform the image data and labels by scaling the input image to 1/255, normalizing it according to the mean value of 0.1307 and standard deviation value of 0.3081, and then packing the processed dataset into a batch of size 64.

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
# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)
```

The dataset can be accessed iteratively using [create_tuple_iterator](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_tuple_iterator.html) or [create_dict_iterator](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_dict_iterator.html) to see the shape and datatype of the data and labels.

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

## Model Building

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

## Model Training

In model training, a complete training process (STEP) requires the realization of the following three steps:

1. **Forward computation**: the model predicts the results, and with the correct label to find the predicted loss.
2. **Backpropagation**: automatically solves for the gradients of the model parameters with respect to the loss, using an automatic differentiation mechanism.
3. **Parameter optimization**: update the gradients to the parameters.

MindSpore uses a functional automatic differentiation mechanism, so for the above steps need to be implemented:

1. Define the forward computation function.
2. Use [value_and_grad](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.value_and_grad.html) to obtain the gradient computation function by functional transformation.
3. Define the training function and use [set_train](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.set_train) to set to training mode, perform forward computation, backpropagation and parameter optimization.

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

In addition to training, we define test functions to evaluate the performance of the model.

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

The training process requires multiple iterations of the dataset, and a complete iteration is called a round (epoch). In each round, the training set is traversed for training and at the end the test set is used for prediction. Printing the loss value and prediction accuracy for each round, we can see that the loss is decreasing and Accuracy is increasing.

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
loss: 2.302898  [  0/938]
loss: 1.729961  [100/938]
loss: 0.865714  [200/938]
loss: 0.782822  [300/938]
loss: 0.389282  [400/938]
loss: 0.293149  [500/938]
loss: 0.474819  [600/938]
loss: 0.242542  [700/938]
loss: 0.542277  [800/938]
loss: 0.342929  [900/938]
Test:
 Accuracy: 90.7%, Avg loss: 0.321954

Epoch 2
-------------------------------
loss: 0.249492  [  0/938]
loss: 0.347967  [100/938]
loss: 0.220382  [200/938]
loss: 0.308149  [300/938]
loss: 0.353044  [400/938]
loss: 0.392116  [500/938]
loss: 0.396438  [600/938]
loss: 0.231412  [700/938]
loss: 0.194819  [800/938]
loss: 0.228290  [900/938]
Test:
 Accuracy: 93.0%, Avg loss: 0.249993

Epoch 3
-------------------------------
loss: 0.343888  [  0/938]
loss: 0.307786  [100/938]
loss: 0.153425  [200/938]
loss: 0.254917  [300/938]
loss: 0.198072  [400/938]
loss: 0.108963  [500/938]
loss: 0.202033  [600/938]
loss: 0.340418  [700/938]
loss: 0.144911  [800/938]
loss: 0.175447  [900/938]
Test:
 Accuracy: 93.7%, Avg loss: 0.212180

Done!
```

## Saving Models

Once the model is trained, its parameters need to be saved.

```python
# Save checkpoint
mindspore.save_checkpoint(model, "model.ckpt")
print("Saved Model to model.ckpt")
```

```text
Saved Model to model.ckpt
```

## Loading Weights

Loading the saved weights is a two-step process:

1. re-instantiate the model object and construct the model.
2. load the model parameters and load them onto the model.

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

> `param_not_load` is the list of parameters that were not loaded. Being empty means all parameters were loaded successfully.

## Model Inference

The loaded model can be used directly for predictive inference.

```python
import matplotlib.pyplot as plt

model.set_train(False)
for data, label in test_dataset:
    pred = model(data)
    predicted = pred.argmax(1)
    print(f'Predicted: "{predicted[:6]}", Actual: "{label[:6]}"')

    # Display the number and the predicted value of the number
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # If the prediction is correct, it will be displayed in blue; if the prediction is incorrect, it will be displayed in red
        color = 'blue' if predicted[i] == label[i] else 'red'
        plt.title('Predicted:{}'.format(predicted[i]), color=color)
        plt.imshow(data.asnumpy()[i][0], interpolation="None", cmap="gray")
        plt.axis('off')
    plt.show()
    break
```

```text
Predicted: "[2 1 0 4 1 7]", Actual: "[2 1 0 4 1 7]"
```

More examples of MindSpore-based OrangePi development boards are detailed in: [GitHub link](https://github.com/mindspore-courses/orange-pi-mindspore)

The required environment for the operation of this case:

| OrangePi AIpro | Image | CANN Toolkit/Kernels | MindSpore |
| :----:| :----: | :----:| :----: |
| 8T 16G | Ubuntu | 8.0.RC3.alpha002| 2.4.10 |
| 8T 16G | Ubuntu | 8.0.0beta1| 2.5.0 |