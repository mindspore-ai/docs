<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/train.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || **Train** || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Infer](https://www.mindspore.cn/tutorials/en/master/beginner/infer.html)

# Model Training

Model training is generally divided into four steps:

1. Build a dataset.
2. Define a neural network model.
3. Define hyperparameters, a loss function, and an optimizer.
4. Input dataset for training and evaluation.

After we have the dataset and the model, we can train and evaluate the model.

## Necessary Prerequisites

First load the previous code from [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) and [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) to load the previous code.

```python
from mindspore import nn
from mindspore import ops
from mindvision import dataset
from mindspore.dataset import vision

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

train_dataset = training_data.dataset
test_dataset = test_data.dataset

transforms = [
    vision.Rescale(1.0 / 255.0, 0),
    vision.Normalize(mean=(0.1307,), std=(0.3081,)),
    vision.HWC2CHW()
]

train_dataset = train_dataset.map(transforms, 'image').batch(64)
test_dataset = test_dataset.map(transforms, 'image').batch(64)

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

## Hyperparameter

Hyperparameters can be adjusted to control the model training and optimization process. Different hyperparameter values may affect the model training and convergence speed. Currently, deep learning models are optimized by using the batch stochastic gradient descent algorithm. The principle of the stochastic gradient descent algorithm is as follows:

$$w_{t+1}=w_{t}-\eta \frac{1}{n} \sum_{x \in \mathcal{B}} \nabla l\left(x, w_{t}\right)$$

In the formula, $n$ is the batch size, and $η$ is a learning rate. In addition, $w_{t}$ is the weight parameter in the training batch t, and $\nabla l$ is the derivative of the loss function. In addition to the gradient itself, the two factors directly determine the weight update of the model. From the perspective of the optimization itself, the two factors are the most important parameters that affect the convergence of the model performance. Generally, the following hyperparameters are defined for training:

- **Epoch**: specifies number of times that the dataset is traversed during training.

- **Batch size**: specifies the size of each batch of data to be read. If the batch size is too small, it takes a long time and the gradient oscillation is serious, which is unfavorable to convergence. If the batch size is too large, the gradient directions of different batches do not change, and the local minimum value is easy to fall into. Therefore, you need to select a proper batch size to effectively improve the model precision and global convergence.

- **Learning rate**: If the learning rate is low, the convergence speed slows down. If the learning rate is high, unpredictable results such as no training convergence may occur. Gradient descent is widely used in parameter optimization algorithms for minimizing model errors. The gradient descent estimates the parameters of the model by iterating and minimizing the loss function at each step. The learning rate is used to control the learning progress of a model during iteration.

```python
epochs = 10
batch_size = 32
learning_rate = 1e-2
```

## Training Process

Once the hyperparameters are set, we can loop the input data to train the model. A complete iterative loop of a data set is called an epoch. Each epoch of performing training consists of two steps.

1. Training: iterate over the training dataset and try to converge to the best parameters.
2. Validation/Testing: iterate over the test dataset to check if model performance improves.

### Loss Function

The loss function is used to evaluate the error between the model's predictions (logits) and targets (targets). When training a model, a randomly initialized neural network model starts to predict the wrong results. The loss function evaluates how different the predicted results are from the targets, and the goal of model training is to reduce the error obtained by the loss function.

Common loss functions include `nn.MSELoss` (mean squared error) for regression tasks and `nn.NLLLoss` (negative log-likelihood) for classification. `nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss` to normalize logits and calculate prediction errors.

```python
loss_fn = nn.CrossEntropyLoss()
```

## Optimizer

Model optimization is the process of adjusting the model parameters at each training step to reduce model error, and MindSpore offers several implementations of optimization algorithms called Optimizers. The optimizer internally defines the parameter optimization process of the model (i.e., how the gradient is updated to the model parameters), and all optimization logic is encapsulated in the optimizer object. Here, we use the SGD (Stochastic Gradient Descent) optimizer.

We obtain the trainable parameters of the model via the `model.trainable_params()` method and pass in the learning rate hyperparameter to initialize the optimizer.

```python
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)
```

In the training process, the gradient corresponding to the parameters can be calculated by the differentiation function, which can be passed into the optimizer to achieve parameter optimization. The form is as follows:

```python
grads = grad_fn(inputs)
optimizer(grads)
```

### Implementing Training and Evaluation

Next, we define the `train_loop` function for training and the `test_loop` function for testing.

To use functional automatic differentiation, we need to define the forward function `forward_fn` and use `ops.value_and_grad` to obtain the differentiation function `grad_fn`. Then, we encapsulate the execution of the differentiation function and the optimizer into the `train_step` function, and then just iterate through the dataset for training.

```python
def train_loop(model, dataset, loss_fn, optimizer):
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
        correct += (pred.argmax(1) == label).sum().asnumpy()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

We pass the instantiated loss function and optimizer into `train_loop` and `test_loop`, train it for 3 rounds and output loss and Accuracy to see the performance change.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)

epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataset, loss_fn, optimizer)
    test_loop(model, test_dataset, loss_fn)
print("Done!")
```

```text
Output exceeds the size limit. Open the full output data in a text editor

Epoch 1
-------------------------------
loss: 2.303159  [  0/938]
loss: 2.288858  [100/938]
loss: 2.265319  [200/938]
loss: 2.181137  [300/938]
loss: 1.851541  [400/938]
loss: 1.529397  [500/938]
loss: 1.004398  [600/938]
loss: 0.843776  [700/938]
loss: 0.507560  [800/938]
loss: 0.539427  [900/938]
Test:
 Accuracy: 84.9%, Avg loss: 0.525360

Epoch 2
-------------------------------
loss: 0.794308  [  0/938]
loss: 0.409702  [100/938]
loss: 0.686628  [200/938]
loss: 0.570685  [300/938]
loss: 0.353630  [400/938]
loss: 0.333396  [500/938]
loss: 0.379772  [600/938]
loss: 0.322043  [700/938]
loss: 0.233923  [900/938]
Test:
 Accuracy: 91.9%, Avg loss: 0.279896

Done!
```
