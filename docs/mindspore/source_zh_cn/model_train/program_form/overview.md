# 编程形态概述

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/program_form/overview.md)

## 静态图和动态图的概念

目前主流的深度学习框架的执行模式有两种，分别为静态图模式和动态图模式。

静态图模式下，程序在编译执行时先生成神经网络的图结构，然后再执行图中涉及的计算操作。因此，在静态图模式下，编译器利用图优化等技术对执行图进行更大程度的优化，从而获得更好的执行性能，有助于规模部署和跨平台运行。

动态图模式下，程序按照代码的编写顺序执行，在执行正向过程中根据反向传播的原理，动态生成反向执行图。这种模式下，编译器将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。

## MindSpore静态图

在MindSpore中，静态图模式又被称为Graph模式，可以通过`set_context(mode=GRAPH_MODE)`来设置成静态图模式。静态图模式比较适合网络固定且需要高性能的场景。在静态图模式下，基于图优化、计算图整图下沉等技术，编译器可以针对图进行全局的优化，因此在静态图下能获得较好的性能，但是执行图是从源码转换而来，因此在静态图下不是所有的Python语法都能支持。

### Graph模式执行原理

在Graph模式下，MindSpore通过源码转换的方式，将Python的源码转换成IR，再在此基础上进行相关的图优化，最终在硬件设备上执行优化后的图。MindSpore使用的是一种基于图表示的函数式IR，即MindIR，采用了接近于ANF函数式的语义。Graph模式是基于MindIR进行编译优化的，使用Graph模式时，需要使用`nn.Cell`类并且在`construct`函数中编写执行代码，或者调用`@jit`装饰器。

### Graph模式自动微分原理

在MindSpore中，Graph模式下的自动微分原理可以参考[自动微分](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/autograd.html)。

## MindSpore动态图

在MindSpore中，动态图模式又被称为PyNative模式，可以通过`set_context(mode=PYNATIVE_MODE)`来设置成动态图模式。在脚本开发和网络流程调试中，推荐使用动态图模式进行调试，支持执行单算子、普通函数和网络、以及单独求梯度的操作。

### PyNative模式执行原理

在PyNative模式下，用户可以使用完整的Python API，此外针对使用MindSpore提供的API时，框架会根据用户选择的硬件平台（Ascend，GPU，CPU），将算子API的操作在对应的硬件平台上执行，并返回相应的结果。框架整体的执行过程如下：

![process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/framework.png)

此例中，当调用到Python接口ops.add(x, y)时，会将Python的接口调用通过Pybind11调用到框架的C++层，转换成C++的调用，接着框架会根据用户设置的device_target选择对应的硬件设备，在该硬件设备上执行add这个操作。

从上述原理可以看到，在PyNative模式下，Python脚本代码会根据Python的语法进行执行，而执行过程中涉及到MindSpore的API，会根据用户设置在不同的硬件上进行执行，从而进行加速。因此，在PyNative模式下，用户可以随意使用Python的语法以及调试方法。例如可以使用常见的PyCharm、VS Code等IDE进行代码的调试。

### PyNative模式自动微分原理

在前面的介绍中，我们可以看出，在PyNative下执行正向过程完全是按照Python的语法进行执行。在PyNative下是基于Tensor进行实现反向传播的，我们在执行正向过程中，将所有应用于Tensor的操作记录下来，并针对每个操作求取其反向，并将所有反向过程串联起来形成整体反向传播图（简称反向图）。最终，将反向图在设备上进行执行计算出梯度。

![forward](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/forward.png)

![backward](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/backward.png)

可以看到对Mul的输入求反向，需要两个输入和输出的反向传播梯度值，此时根据实际的输入值，可以将z连接到MulGrad。以此类推，对下一个算子Matmul，相应的得到MatmulGrad信息，再根据bprop的输入输出，将上下文梯度传播连接起来。

同理对于输入y求导，可以使用同样的过程进行推导。

### PyNative模式下的控制流

在PyNative模式下，脚本按照Python的语法执行，因此在MindSpore中，针对控制流语法并没有做特殊处理，直接按照Python的语法直接展开执行，进而对展开的执行算子进行自动微分的操作。例如，对于for循环，在PyNative下会根据具体的循环次数，不断的执行for循环中的语句，并对其算子进行自动微分的操作。
