<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || **Model** || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Infer](https://www.mindspore.cn/tutorials/en/master/beginner/infer.html)

# Building a Network

The neural network model consists of neural network layers and Tensor operations. `mindspore.nn` provides common neural network layer implementations, and the `Cell` class in MindSpore is the base class for building all networks and is the basic unit of the network. `Cell`, a neural network model, is composed of different sub-`Cells`. Using such a nested structure, the neural network structure can be constructed and managed simply by using object-oriented programming thinking.

In the following we will construct a neural network model for the Mnist dataset classification.

```python
import mindspore
from mindspore import nn, ops
```

## Defining a Model Class

When define a neural network, we can inherit the `nn.Cell` class, instantiate and manage the state of the sub-Cell in the `__init__` method, and implement the Tensor operation in the `construct` method.

> `construct` means neural network (computational graph) construction. For more details, see [computational graph](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html).

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
```

After completing construction, instantiate the `Network` object and look at its structure.

```python
model = Network()
model
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

We construct an input data and call the model directly to obtain a 10-dimensional Tensor output that contains the original predicted values for each category.

> The `model.construct()` method cannot be called directly.

```python
X = ops.ones((1, 28, 28), mindspore.float32)
logits = model(X)
logits
```

```text
Tensor(shape=[1, 10], dtype=Float32, value=
[[-5.08734025e-04,  3.39190010e-04,  4.62840870e-03 ... -1.20305456e-03, -5.05689112e-03,  3.99264274e-03]])
```

On this basis, we obtain the prediction probabilities by an `nn.Softmax` layer instance.

```python
pred_probab = nn.Softmax(axis=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

```text
Predicted class: [4]
```

## Model Layers

In this section, we decompose each layer of the neural network model constructed in the previous section. First we construct a random data (3 images of 28x28) with shape (3, 28, 28) and pass through each neural network layer in turn to observe its effect.

```python
input_image = ops.ones((3, 28, 28), mindspore.float32)
input_image.shape
```

```text
(3, 28, 28)
```

### nn.Flatten

Initialize the `nn.Flatten` layer and convert a 28x28 2D tensor into a contiguous array of size 784.

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
flat_image.shape
```

```text
(3, 784)
```

### nn.Dense

`nn.Dense` is the fully connected layer, which linearly transforms the input by using weights and deviations.

```python
layer1 = nn.Dense(in_channels=28*28, out_channels=20)
hidden1 = layer1(flat_image)
hidden1.shape
```

```text
(3, 20)
```

### nn.ReLU

`nn.ReLU` layer adds a nonlinear activation function to the network, to help the neural network learn various complex features.

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

```text
Before ReLU: [[-0.04736331  0.2939465  -0.02713677 -0.30988005 -0.11504349 -0.11661264
   0.18007928  0.43213072  0.12091967 -0.17465964  0.53133243  0.12605792
   0.01825903  0.01287796  0.17238477 -0.1621131  -0.0080034  -0.24523425
  -0.10083733  0.05171938]
 [-0.04736331  0.2939465  -0.02713677 -0.30988005 -0.11504349 -0.11661264
   0.18007928  0.43213072  0.12091967 -0.17465964  0.53133243  0.12605792
   0.01825903  0.01287796  0.17238477 -0.1621131  -0.0080034  -0.24523425
  -0.10083733  0.05171938]
 [-0.04736331  0.2939465  -0.02713677 -0.30988005 -0.11504349 -0.11661264
   0.18007928  0.43213072  0.12091967 -0.17465964  0.53133243  0.12605792
   0.01825903  0.01287796  0.17238477 -0.1621131  -0.0080034  -0.24523425
  -0.10083733  0.05171938]]


After ReLU: [[0.         0.2939465  0.         0.         0.         0.
  0.18007928 0.43213072 0.12091967 0.         0.53133243 0.12605792
  0.01825903 0.01287796 0.17238477 0.         0.         0.
  1.         0.05171938]
 [0.         0.2939465  0.         0.         0.         0.
  0.18007928 0.43213072 0.12091967 0.         0.53133243 0.12605792
  0.01825903 0.01287796 0.17238477 0.         0.         0.
  1.         0.05171938]
 [0.         0.2939465  0.         0.         0.         0.
  0.18007928 0.43213072 0.12091967 0.         0.53133243 0.12605792
  0.01825903 0.01287796 0.17238477 0.         0.         0.
  1.         0.05171938]]
```

### nn.SequentialCell

`nn.SequentialCell` is an ordered Cell container. The input Tensor will pass through all the Cells in the defined order, and we can use `SequentialCell` to construct a neural network model quickly.

```python
seq_modules = nn.SequentialCell(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Dense(20, 10)
)

logits = seq_modules(input_image)
logits.shape
```

```text
(3, 10)
```

### nn.Softmax

Finally, the value of logits returned by the last fully-connected layer of the neural network is scaled to \[0, 1\] by using `nn.Softmax`, indicating the predicted probability of each category. The dimensional values specified by `axis` sum to 1.

```python
softmax = nn.Softmax(axis=1)
pred_probab = softmax(logits)
```

## Model Parameters

The internal neural network layer of the network has weight parameters and bias parameters (e.g. `nn.Dense`), which are continuously optimized during the training process, and the parameter names and corresponding parameter details can be obtained through `model.parameters_and_names()`.

```python
print(f"Model structure: {model}\n\n")

for name, param in model.parameters_and_names():
    print(f"Layer: {name}\nSize: {param.shape}\nValues : {param[:2]} \n")
```

```text
Output exceeds the size limit. Open the full output data in a text editor

Model structure: Network<
  (flatten): Flatten<>
  (dense_relu_sequential): SequentialCell<
    (0): Dense<input_channels=784, output_channels=512, has_bias=True>
    (1): ReLU<>
    (2): Dense<input_channels=512, output_channels=512, has_bias=True>
    (3): ReLU<>
    (4): Dense<input_channels=512, output_channels=10, has_bias=True>
    >
  >


Layer: dense_relu_sequential.0.weight
Size: (512, 784)
Values : [[-0.01491369  0.00353318 -0.00694948 ...  0.01226766 -0.00014423
   0.00544263]
 [ 0.00212971  0.0019974  -0.00624789 ... -0.01214037  0.00118004
  -0.01594325]]

Layer: dense_relu_sequential.0.bias
Size: (512,)
Values : [0. 0.]

Layer: dense_relu_sequential.2.weight
Size: (512, 512)

Layer: dense_relu_sequential.4.bias
Size: (10,)
Values : [0. 0.]
```

For more built-in neural network layers, see [mindspore.nn API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html).
