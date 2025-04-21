[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/train.md)

[Introduction](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/autograd.html) || **Train** || [Save and Load](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/mixed_precision.html)

# Model Training

Model training is generally divided into four steps:

1. Build a dataset.
2. Define a neural network model.
3. Define hyperparameters, a loss function, and an optimizer.
4. Input dataset for training and evaluation.

After we have the dataset and the model, we can train and evaluate the model.

## Building a Dataset

First load the previous code from [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) to build a dataset.

```python
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset

# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)


def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe('MNIST_Data/train', batch_size=64)
test_dataset = datapipe('MNIST_Data/test', batch_size=64)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:05<00:00, 2.07MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

## Defining a Neural Network Model

Load the code from [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) to define a neural network model.

```python
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
```

## Defining the Hyperparameter, Loss Function and Optimizer

### Hyperparameter

Hyperparameters can be adjusted to control the model training and optimization process. Different hyperparameter values may affect the model training and convergence speed. Currently, deep learning models are optimized by using the batch stochastic gradient descent algorithm. The principle of the stochastic gradient descent algorithm is as follows:

$$w_{t+1}=w_{t}-\eta \frac{1}{n} \sum_{x \in \mathcal{B}} \nabla l\left(x, w_{t}\right)$$

In the formula, $n$ is the batch size, and $η$ is a learning rate. In addition, $w_{t}$ is the weight parameter in the training batch t, and $\nabla l$ is the derivative of the loss function. In addition to the gradient itself, the two factors directly determine the weight update of the model. From the perspective of the optimization itself, the two factors are the most important parameters that affect the convergence of the model performance. Generally, the following hyperparameters are defined for training:

- **Epoch**: specifies number of times that the dataset is traversed during training.

- **Batch size**: specifies the size of each batch of data to be read. If the batch size is too small, it takes a long time and the gradient oscillation is serious, which is unfavorable to convergence. If the batch size is too large, the gradient directions of different batches do not change, and the local minimum value is easy to fall into. Therefore, you need to select a proper batch size to effectively improve the model precision and global convergence.

- **Learning rate**: If the learning rate is low, the convergence speed slows down. If the learning rate is high, unpredictable results such as no training convergence may occur. Gradient descent is widely used in parameter optimization algorithms for minimizing model errors. The gradient descent estimates the parameters of the model by iterating and minimizing the loss function at each step. The learning rate is used to control the learning progress of a model during iteration.

```python
epochs = 3
batch_size = 64
learning_rate = 1e-2
```

### Loss Function

The loss function is used to evaluate the error between the model's predictions (logits) and targets (targets). When training a model, a randomly initialized neural network model starts to predict the wrong results. The loss function evaluates how different the predicted results are from the targets, and the goal of model training is to reduce the error obtained by the loss function.

Common loss functions include `nn.MSELoss` (mean squared error) for regression tasks and `nn.NLLLoss` (negative log-likelihood) for classification. `nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss` to normalize logits and calculate prediction errors.

```python
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer

Model optimization is the process of adjusting the model parameters at each training step to reduce model error, and MindSpore offers several implementations of optimization algorithms called Optimizers. The optimizer internally defines the parameter optimization process of the model (i.e., how the gradient is updated to the model parameters), and all optimization logic is encapsulated in the optimizer object. Here, we use the SGD (Stochastic Gradient Descent) optimizer.

We obtain the trainable parameters of the model via the `model.trainable_params()` method and pass in the learning rate hyperparameter to initialize the optimizer.

```python
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)
```

> In the training process, the gradient corresponding to the parameters can be calculated by the differentiation function, which can be passed into the optimizer to achieve parameter optimization. The form is as follows:
>
> grads = grad_fn(inputs)
>
> optimizer(grads)

## Training and Evaluation

Once the hyperparameters, loss function and optimizer are set, we can loop the input data to train the model. A complete iterative loop of a data set is called an epoch. Each epoch of performing training consists of two steps.

1. Training: iterate over the training dataset and try to converge to the best parameters.
2. Validation/Testing: iterate over the test dataset to check if model performance improves.

Next, we define the `train_loop` function for training and the `test_loop` function for testing.

To use functional automatic differentiation, we need to define the forward function `forward_fn` and use [value_and_grad](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.value_and_grad.html) to obtain the differentiation function `grad_fn`. Then, we encapsulate the execution of the differentiation function and the optimizer into the `train_step` function, and then just iterate through the dataset for training.

```python
# Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train_loop(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
```

The `test_loop` function also traverses the dataset, calls the model to calculate the loss and Accuracy and returns the final result.

```python
def test_loop(model, dataset, loss_fn):
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

We pass the instantiated loss function and optimizer into `train_loop` and `test_loop`, train it for 3 rounds and output loss and Accuracy to see the performance change.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataset)
    test_loop(model, test_dataset, loss_fn)
print("Done!")
```

```text
Epoch 1
-------------------------------
loss: 2.302806  [  0/938]
loss: 2.285086  [100/938]
loss: 2.264712  [200/938]
loss: 2.174010  [300/938]
loss: 1.931853  [400/938]
loss: 1.340721  [500/938]
loss: 0.953515  [600/938]
loss: 0.756860  [700/938]
loss: 0.756263  [800/938]
loss: 0.463846  [900/938]
Test:
 Accuracy: 84.7%, Avg loss: 0.527155

Epoch 2
-------------------------------
loss: 0.479126  [  0/938]
loss: 0.437443  [100/938]
loss: 0.685504  [200/938]
loss: 0.395121  [300/938]
loss: 0.550566  [400/938]
loss: 0.459457  [500/938]
loss: 0.293049  [600/938]
loss: 0.422102  [700/938]
loss: 0.333153  [800/938]
loss: 0.412182  [900/938]
Test:
 Accuracy: 90.5%, Avg loss: 0.335083

Epoch 3
-------------------------------
loss: 0.207366  [  0/938]
loss: 0.343559  [100/938]
loss: 0.391145  [200/938]
loss: 0.317566  [300/938]
loss: 0.200746  [400/938]
loss: 0.445798  [500/938]
loss: 0.603720  [600/938]
loss: 0.170811  [700/938]
loss: 0.411954  [800/938]
loss: 0.315902  [900/938]
Test:
 Accuracy: 91.9%, Avg loss: 0.279034

Done!
```
