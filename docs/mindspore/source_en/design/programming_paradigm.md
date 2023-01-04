# Programming Paradigm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/programming_paradigm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Programming paradigm refers to the programming style or programming approach of a programming language. Typically, AI frameworks rely on the programming paradigm of the programming language used by the front-end programming interface for the construction and training of neural networks. MindSpore, as an AI+scientific computing convergence computing framework, provides object-oriented programming and functional programming support for AI and scientific computing scenarios, respectively. At the same time, in order to enhance the flexibility and ease of use of the framework, a functional + object-oriented fusion programming paradigm is proposed, which effectively reflects the advantages of functional automatic differentiation mechanism.

The following describes each of the three types of programming paradigms supported by MindSpore and their simple examples.

## Object-oriented Programming

Object-oriented programming (OOP) is a programming method that decomposes programs into modules (classes) that encapsulate data and related operations, with objects being instances of classes. Object-oriented programming uses objects as the basic unit of a program, encapsulating the program and data in order to improve the reusability, flexibility and extensibility of the software, and the program in the object can access and often modify the data associated with the object.

In a general programming scenario, code and data are the two core components. Object-oriented programming is to design data structures for specific objects to define classes (Class). The class usually consists of the following two parts, corresponding to code and data, respectively:

- Methods
- Attributes

For different objects obtained after the instantiation of the same Class, the methods and attributes are the same, but the difference is the values of the attributes. The different attribute values determine the internal state of the object, so OOP can be good for state management.

The following is an example of a simple class constructed in Python:

```python
class Sample: #class declaration
    def __init__(self, name): # class constructor (code)
        self.name = name # attribute (data)

    def set_name(self, name): # method declaration (code)
        self.name = name # method implementation (code)
```

For constructing a neural network, the primary component is the network layer (Layer), and a neural network layer contains the following components:

- Tensor Operation
- Weights

These two correspond exactly to the Methods and Attributes of the class, and the weights themselves are the internal states of the neural network layer, so using the class to construct Layers naturally fits its definition. In addition, we wish to use the neural network layer for stacking and construct deep neural networks when programming, and new Layer classes can be easily constructed by combining Layer objects using OOP programming. In addition, we wish to use neural network layers for stacking and constructing deep neural networks when programming, and new Layer classes can be easily constructed by combining Layer objects by using OOP programming.

The following is an example of a neural network class constructed by using MindSpore:

```python
from mindspore import nn, Parameter
from mindspore.common.initializer import initializer

class Linear(nn.Cell):
    def __init__(self, in_features, out_features, has_bias): # class constructor (code)
        super().__init__()
        self.weight = Parameter(initializer('normal', [out_features, in_features], mindspore.float32), 'weight') # layer weight (data)
        self.bias = Parameter(initializer('zeros', [out_features], mindspore.float32), 'bias') # layer weight (data)

    def construct(self, inputs): # method declaration (code)
        output = ops.matmul(inputs, self.weight.transpose(0, 1)) # tensor transformation (code)
        output = output + self.bias # tensor transformation (code)
        return output
```

In addition to the construction of the neural network layer by using the object-oriented programming paradigm, MindSpore supports pure object-oriented programming to construct the neural network training logic, where the forward computation, back propagation, gradient optimization and other operations of the neural network are constructed by using classes. The following is an example of pure object-oriented programming.

```python
import mindspore
import mindspore.nn as nn
from mindspore import value_and_grad

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad_fn = value_and_grad(self.network, None, self.optimizer.parameters)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        self.optimizer(grads)
        return loss

network = nn.Dense(5, 3)
loss_fn = nn.BCEWithLogitsLoss()
network_with_loss = nn.WithLossCell(network, loss_fn)
optimizer = nn.SGD(network.trainable_params(), 0.001)
trainer = TrainOneStepCell(network_with_loss, optimizer)
```

At this point, both the neural network and its training process are managed by using classes that inherit from `nn.Cell`, which can be easily compiled and accelerated as a computational graph.

## Functional Programming

Functional programming is a programming paradigm that treats computer operations as functions and avoids the use of program state and mutable objects.

In the functional programming, functions are treated as first-class citizens, which means they can be bound to names (including local identifiers), passed as arguments, and returned from other functions, just like any other data type. This allows programs to be written in a declarative and composable style, where small functions are combined in a modular fashion. Functional programming is sometimes seen as synonymous with pure functional programming, a subset of functional programming that treats all functions as deterministic mathematical functions or pure functions. When a pure function is called with some given parameters, it will always return the same result and is not affected by any mutable state or other side effects.

The functional programming has two core features that make it well suited to the needs of scientific computing:

1. The programming function semantics are exactly equivalent to the mathematical function semantics.
2. Determinism, if the same input is given, the same output is returned. No side effects.

Due to this feature of determinism, by limiting side effects, programs can have fewer errors, are easier to debug and test, and are more suitable for formal verification.

MindSpore provides pure functional programming support. With the numerical computation interfaces provided by `mindspore.numpy` and `mindspore.scipy`, you can easily program scientific computations. The following is an example of using functional programming:

```python
import mindspore.numpy as mnp
from mindspore import grad

grad_tanh = grad(mnp.tanh)
print(grad_tanh(2.0))
# 0.070650816

print(grad(grad(mnp.tanh))(2.0))
print(grad(grad(grad(mnp.tanh)))(2.0))
# -0.13621868
# 0.25265405
```

In line with the needs of the functional programming paradigm, MindSpore provides a variety of functional transformation interfaces, including automatic differentiation, automatic vectorization, automatic parallelism, just-in-time compilation, data sinking and other functional modules, which are briefly described below:

- Automatic differentiation: `grad`, `value_and_grad`, providing differential function transformation.
- Automatic vectorization: A higher-order function for mapping a function fn along the parameter axis.
- Automatic parallelism: `shard`, a functional operator slice, specifying the distribution strategy of the function input/output Tensor.
- Just-in-time compilation: `jit`, which compiles a Python function into a callable MindSpore graph.
- Data sinking: `data_sink`, transform the input function to obtain a function that can use the data sink pattern.

Based on the above function transformation interfaces, function transformations can be used quickly and efficiently to implement complex functions when using the functional programming paradigm.

## Functional + Object-Oriented Fusion Programming

Taking into account the flexibility and ease of use of the neural network model construction and training process, combined with MindSpore's own functional automatic differentiation mechanism, MindSpore has designed a functional + object-oriented fusion programming paradigm for AI model training, which can combine the advantages of object-oriented programming and functional programming. The same set of automatic differentiation mechanism is also used to achieve the compatibility of deep learning back propagation and scientific computing automatic differentiation, supporting the compatibility of AI and scientific computing modeling from the bottom. The following is a typical process for the functional + object-oriented fusion programming:

1. Constructing neural networks with classes.
2. Instantiating neural network objects.
3. Constructing the forward function, and connecting the neural network and the loss function.
4. Using function transformations to obtain gradient calculation (back propagation) functions.
5. Constructing training process functions.
6. Calling functions for training.

The following is a simple example of functional + object-oriented fusion programming:

```python
# Class definition
class Net(nn.Cell):
    def __init__(self):
        ......
    def construct(self, inputs):
        ......

# Object instantiation
net = Net() # network
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = nn.Adam(net.trainable_params(), lr) # optimizer

# define forward function
def forword_fn(inputs, targets):
    logits = net(inputs)
    loss = loss_fn(logits, targets)
    return loss, logits

# get grad function
grad_fn = value_and_grad(forward_fn, None, optim.parameters, has_aux=True)

# define train step function
def train_step(inputs, targets):
    (loss, logits), grads = grad_fn(inputs, targets) # get values and gradients
    optimizer(grads) # update gradient
    return loss, logits

for i in range(epochs):
    for inputs, targets in dataset():
        loss = train_step(inputs, targets)
```

As in the above example, object-oriented programming is used in the construction of the neural network, and the neural network layers are constructed in a manner consistent with the conventions of AI programming. When performing forward computation and backward propagation, MindSpore uses functional programming to construct the forward computation as a function, then obtain `grad_fn` by function transformation, and finally obtain the gradient corresponding to the weights by executing `grad_fn`.

The functional + object-oriented fusion programming  ensures the ease of use of neural network construction and improves the flexibility of training processes such as forward computation and backward propagation, which is the default programming paradigm recommended by MindSpore.
