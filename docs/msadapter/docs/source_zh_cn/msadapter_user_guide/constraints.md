# MSAadpter机制性约束

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/msadapter/docs/source_zh_cn/msadapter_user_guide/constraints.md)

本文介绍MindSpore和PyTorch实现上的主要区别：

1. 微分机制
2. Dispatch机制
3. Storage机制
4. 静态编译机制

这些机制的不同，导致用户需要手动修改代码，或者部分功能暂时不支持。

## 微分机制

PyTorch 使用的是动态计算图，运算在代码执行时立即计算，正反向计算图在每次前向传播时动态构建。PyTorch微分为命令式反向微分，更符合面对对象编程的使用习惯。

MindSpore使用函数式自动微分的设计理念，提供更接近于数学语义的自动微分接口。MindSpore的微分可以简单理解为函数包装了一个模型，接口是一个函数，而PyTorch的接口是模型。

详情参考[MindSpore函数式微分](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/autograd.html)。

由于两者底层设计的不同，MSAdapter不支持以下相关接口：

- torch.Tensor.backward()
- torch.autograd.*

部分相关接口PyTorch原始用法如下：

```python
loss.backward()
```

```python
loss.backward(gradient_tensor)
```

由于计算图的实现差异（包含命令式微分与函数式微分的差异），**原始正向计算以及反向计算需要用户手动进行如下修改**：

原始PyTorch写法：

```python
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
```

MSAdapter修改后的写法：

```python
import mindspore
def forward_fn(inputs, labels):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    return loss
grad_fn = mindspore.value_and_grad(forward_fn, None, weights=model.trainable_params())

loss, grads = grad_fn(inputs, labels)
```

mindspore.value_and_grad的第二个参数为grad_position，定义对于哪些输入进行求导。当用户设置为None的时候必须定义weights。详细定义可以参考[mindspore.value_and_grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.value_and_grad.html)。

MSAdapter只需要调用grad_fn就能够一次性调用正向计算加反向计算。

## Dispatch机制

`torch.dispatch` 是 PyTorch 中用于拦截和自定义张量操作的机制。它允许开发者在张量运算被调用时插入自己的逻辑，无论是为了调试、性能优化，还是实现特定领域的功能增强。这个特性是在 PyTorch 1.7中引入的，并且为高级用户提供了强大的能力来扩展PyTorch的功能。

- Dispatch机制：当用户执行任何张量操作（例如加法、乘法等）时，实际上都是通过PyTorch的dispatch机制进行的。该机制决定了哪个具体实现（CPU、CUDA等）应该被执行。
- torch.Tensor的子类化：通过创建一个继承自torch.Tensor的新类，并使用@torch_dispatch装饰器，用户可以自定义这些操作的行为。

当前MindSpore Storage机制在规划设计中，未来会支持。当前MSAdapter暂时不支持以下相关接口：

1. torch.Tensor.to()
2. torch自定义算子涉及dispatch部分

**以上接口需要用户手动修改，即无需进行.to()操作， MSAdapter默认将模型与Tensor放置于Ascend NPU。可参考[快速入门](quick_start.md#torchtensorto)。**

## Storage机制

`torch.Storage` 是 PyTorch 中用于存储和管理张量数据的基础类。它是单一类型的数据连续存储区域，是构成 torch.Tensor 的底层组件。每个 Tensor 都有一个对应的 Storage，它实际保存了张量中的元素。

- 单一数据类型：一个 Storage 只能包含一种数据类型（例如 float、int 等）。这有助于优化内存使用和计算效率。
- 连续存储：所有元素在 Storage 中都是按顺序连续存放的，这样有利于高效访问和操作。
- 共享数据：多个 Tensor 可以共享同一个 Storage，这对于需要视图（view）或切片（slice）操作时特别有用，因为它们可以避免数据的重复拷贝。

由于MindSpore暂时不支持storage特性，MSAdapter不支持以下相关接口：

1. torch.TypedStorage
2. torch.untypedStorage
3. torch序列化反序列化（ckpt保存加载，包括safetensors）

部分相关接口PyTorch原始用法如下：

```python
torch.save(x, 'tensor.pt') # 保存张量
torch.save(model, 'full_model.pt')  # 保存模型结构和参数
torch.save(model.state_dict(), 'model_params.pt') # 仅保存参数（轻量级）
```

```python
x_loaded = torch.load('tensor.pt') # 加载张量
model.load_state_dict(torch.load('model_params.pt'))  # 加载参数
```

**MSAdapter已经支持torch.load，使用MindSpore 2.7.0后，MSAdapter将支持torch.save相关功能。**

## 静态编译机制

### TorchDynamo机制的介绍

PyTorch 在其发展历程中引入了多种机制以提升性能和开发者的体验，其中 TorchDynamo 是一项旨在通过非侵入式的方法捕获 PyTorch 程序的图表示(graph representation)，以便于优化和加速执行的技术。它是在 PyTorch 2.0中引入的一项重要特性，致力于提供一种更加高效、灵活的方式来优化模型训练和推理过程。

- 无侵入式的图捕捉：TorchDynamo 能够在不修改用户代码的情况下自动捕捉 PyTorch 模型的计算图。这意味着开发者无需手动重写或注解他们的代码来利用图捕捉的好处。
- 动态形状支持：与传统的静态图捕捉不同，TorchDynamo 支持动态形状，这使得它非常适合处理输入尺寸变化的任务，如自然语言处理(NLP)和图像处理等领域的应用。
- 后端灵活性：TorchDynamo 不仅限于生成一种特定类型的图。它可以将捕捉到的图传递给不同的后端进行进一步优化和执行，例如 TOSA、TVM 和 NNC 等。
- 性能优化：通过图捕捉和后续的优化步骤，TorchDynamo 可以显著提高模型的执行效率，减少内存使用，并加速训练和推理速度。

### MindSpore动静统一的编程介绍

传统AI框架主要有两种编程执行形态，静态图模式和动态图模式。

- 动态图模式，能有效解决静态图的编程门槛高问题，由于程序是按照代码的编写顺序执行，不做整图编译优化，相对性能优化空间较少，特别是面向DSA等专有硬件的优化具有较大挑战。
- 静态图模式，能有效感知神经网络各层算子间的关系，基于编译技术进行有效的编译优化以提升性能。但传统静态图需要开发者感知构图接口，组建或调试网络比较复杂，且难于与常用Python库、自定义Python函数进行穿插使用。

MindSpore基于Python构建神经网络的图结构，相比于传统的静态图模式，能有更易用、更灵活的表达能力。MindSpore创新性的构建源码转换能力，基于Python语句提取AST进行计算图构建，因此可以支持开发者使用的Python原生语法（条件/循环等）和其他操作，如元组（Tuple）、列表（List）以及Lambda表达来构建计算图，并对计算图进行自动微分。所以MindSpore能更好地兼容动态图和静态图的编程接口，在代码层面保持一致，如控制流写法等。

原生Python表达可基于Python控制流关键字，直接使能静态图模式的执行，使得动静态图的编程统一性更高。同时开发者基于MindSpore的接口，可以灵活的对Python代码片段进行动静态图模式控制。即可以将程序局部函数以静态图模式执行（mindspore.jit）而同时其他函数按照动态图模式执行。从而使得在与常用Python库、自定义Python函数进行穿插执行使用时，开发者可以灵活指定函数片段进行静态图优化加速，而不牺牲穿插执行的编程易用性。

由于两者设计上的不同，MSAdapter不支持以下相关接口：

1. torch.fx相关
2. torch.dynamo
3. torch.compile

部分相关接口PyTorch原始用法如下：

```python
compiled_model = torch.compile(model)
```

```python
traced_model = torch.fx.symbolic_trace(model) # 使用 fx.symbolic_trace 追踪计算图
```

**此类接口由于MindSpore与PyTorch底层实现不同，MSAdapter暂不支持。**
