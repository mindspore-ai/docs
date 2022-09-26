<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/quick_start.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || **Quick Start** || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Infer](https://www.mindspore.cn/tutorials/en/master/beginner/infer.html)

# Quick Start: Linear Fitting

This section quickly implements a simple deep learning model through MindSpore APIs. For a deeper understanding of how to use MindSpore, see the reference links provided at the end of each section.

```python
import mindspore
from mindspore import nn
from mindspore import ops
from mindvision import dataset
from mindspore.dataset import vision
```

## Processing a Dataset

MindSpore provides Pipeline-based [Data Engine](https://www.mindspore.cn/docs/zh-CN/master/design/data_engine.html) and achieves efficient data preprocessing through [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) and [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html). In addition, MindSpore provides domain-specific development libraries, such as [Text](https://gitee.com/mindspore/text/), [Vision](https://gitee.com/mindspore/vision/), etc. The development libraries provide encapsulation for a large number of datasets and can be downloaded quickly. In this tutorial, we use Vision to demonstrate the dataset operation.

`mindvision.dataset` module includes various CV datasets, such as Mnist, Cifar, ImageNet, etc. In this tutorial, we use the Mnist dataset and pre-process dataset by using the data transformations provided by `mindspore.dataset`, after automatically downloaded.

```python
# Download training data from open datasets
training_data = dataset.Mnist(
    path="dataset",
    split="train",
    download=True
)

# Download test data from open datasets
test_data = dataset.Mnist(
    path="dataset",
    split="test",
    download=True
)
```

After the data is downloaded, the dataset object is obtained.

```python
train_dataset = training_data.dataset
test_dataset = test_data.dataset
```

Print the names of the data columns contained in the dataset for dataset pre-processing.

```python
train_dataset.column_names
```

```text
['image', 'label']
```

Here we set the batch size to 64 and define the data transformations to be done on image, including `Rescale`, `Normalize`, and `HWC2CHW`.

```python
batch_size = 64

transforms = [
    vision.Rescale(1.0 / 255.0, 0),
    vision.Normalize(mean=(0.1307,), std=(0.3081,)),
    vision.HWC2CHW()
]
```

Dataset in MindSpore uses the Data Processing Pipeline, which requires specifying operations such as map, batch, and shuffle. Here we use `map` to transform the image data in the `'image'` column, and then pack the processed dataset into a batch of size 64.

```python
# Map vision transforms and batch dataset
train_dataset = train_dataset.map(transforms, 'image').batch(batch_size)
test_dataset = test_dataset.map(transforms, 'image').batch(batch_size)
```

Use `create_tuple_iterator` or `create_dict_iterator` to iterate over the dataset.

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

For more detailed information, see [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) and [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html).

## Building Network

`mindspore.nn` class is the base class for building all networks and is the basic unit of the network. When the user needs to customize the network, you can inherit the `nn.Cell` class and override the `__init__` method and the `construct` method. `__init__` contains the definitions of all network layers, and `construct` contains the transformation process of the data ([Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html)) (i.e. the construction process of the [computational graph](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html)).

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

For more detailed information, see [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html).

## Training Model

```python
# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)
```

In model training, a complete training process (step) requires the following three steps:

1. **Forward calculation**: model predicts results (logits) and finds the prediction loss (loss) with the correct label (label).
2. **Backpropagation**: Using an automatic differentiation mechanism, the gradients of the model parameters (parameters) with respect to the loss are automatically found.
3. **Parameter optimization**: update the gradient to the parameter.

MindSpore uses a functional automatic differentiation mechanism, implemented through the steps above:

1. Define forward calculation function.
2. Obtain the gradient calculation function by function transformation.
3. Define training functions, and perform forward computation, back propagation and parameter optimization.

```python
def train(model, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
```

In addition to training, we define test functions that are used to evaluate the performance of the model.

```Python
def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).sum().asnumpy()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process requires several iterations of the dataset, and one complete iteration is called an epoch. In each round, the training set is traversed for training and the test set is used for prediction at the end. The loss value and prediction accuracy (Accuracy) of each round are printed, and it can be seen that the loss is decreasing and Accuracy is increasing.

```python
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset, loss_fn, optimizer)
    test(model, test_dataset, loss_fn)
print("Done!")
```

```text
Epoch 1
-------------------------------
loss: 2.302365  [  0/938]
loss: 2.288243  [100/938]
loss: 2.264786  [200/938]
loss: 2.189968  [300/938]
loss: 1.986960  [400/938]
loss: 1.381596  [500/938]
loss: 1.018606  [600/938]
loss: 0.883643  [700/938]
loss: 0.873870  [800/938]
loss: 0.536063  [900/938]
Test:
 Accuracy: 85.0%, Avg loss: 0.525184

Epoch 2
-------------------------------
loss: 0.492243  [  0/938]
loss: 0.495629  [100/938]
loss: 0.387343  [200/938]
loss: 0.581129  [300/938]
loss: 0.387734  [400/938]
loss: 0.446312  [500/938]
loss: 0.340325  [600/938]
loss: 0.300699  [700/938]
loss: 0.312408  [800/938]
loss: 0.152036  [900/938]
Test:
 Accuracy: 89.9%, Avg loss: 0.343906

Epoch 3
-------------------------------
loss: 0.287898  [  0/938]
loss: 0.241483  [100/938]
loss: 0.361069  [200/938]
loss: 0.493174  [300/938]
loss: 0.440934  [400/938]
loss: 0.234498  [500/938]
loss: 0.234642  [600/938]
loss: 0.229699  [700/938]
loss: 0.221106  [800/938]
loss: 0.474382  [900/938]
Test:
 Accuracy: 91.9%, Avg loss: 0.281938

Done!
```

For the detailed information, see [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html).

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
param_not_load = mindspore.load_param_into_net(model, param_dict)
param_not_load
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
Predicted: "[3 5 4 5 6 1 6 4 0 6]", Actual: "[3 5 9 3 6 1 6 4 0 6]"
```

For more detailed information, see [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html).
