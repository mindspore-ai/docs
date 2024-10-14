# Static Graph Dynamic Shape

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_en/model_train/program_form/static_graph_syntax/dynamic_shape.md)

## Background

Dynamic shape is a common research topic in the field of deep learning frameworks. MindSpore has also done a lot of exploration and research on dynamic shape, and initially supports the ability of dynamic shape in static graph mode based on the results of the research.

This paper focuses on MindSpore static graph dynamic shape for the introduction, and the dynamic shape are generalized static graph dynamic shape.

The core problem that needs to be solved is how to do multiple executions in one compilation when the input data size changes. The flow comparison between processing multi-size data through static shape and processing multi-size data through dynamic shape is illustrated below:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/docs/mindspore/source_zh_cn/design/images/dynamic_shape/static_dynamic_shape_diff.png)

As shown in the above figure, when multiple sizes of data are input, static shapes are compiled once for each input size, whereas dynamic shapes are compiled only once, so dynamic shapes save a lot of compilation time compared to static shapes, and therefore improve the end-to-end execution performance of the network.

## Symbol Engine Design

Although dynamic shapes make up for the lack of multiple compilation in static shapes, they also bring new challenges such as degraded execution performance, inability to perform parallel slicing, and inability to optimize memory reuse.

MindSpore inherits most of the parallel slicing and operator fusion capabilities for static shape scenario, and achieves deep memory optimization through virtual memory to achieve dynamic shape execution performance and memory efficiency up to about 90% of that of static shape.

Dynamic shape expresses shape by symbolic shape, for example, there are two sets of input data as Tensor(shape=(8, 10)) and Tensor(shape=(8, 100)), using static shape compiled many times will produce two kinds of IR, Tensor(shape=(8, 10)) and Tensor(shape=( 8, 100)). Dynamic shape produces Tensor(shape=(8, Any)). Any means axis is dynamic, and the symbolic engine shape can be further expressed dynamic shape IR as Tensor(shape=(8, 10*s1)). Symbolic shape expresses the shape derivation process through symbolic operations to achieve the ability to replace numerical judgments with symbolic judgments in dynamic shape scenarios. An example of one IR based on the symbolic engine to derive a dynamic shape is as follows:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/docs/mindspore/source_zh_cn/design/images/dynamic_shape/symbol_engine.png)

As shown in the figure, the symbol engine labels the input node's shape as `s1`, `s2`, etc., and stores the output shape as an expression based on the input shape when the operator shape is derived. For example, for the `40Mul` node, its output shape is no longer Any but `max(s1,s2)`; for the `104BatchMatMul` node, based on the constraints of matrix multiplication, one can directly set `s4 == s6`; for the `112Add` node, since `s5` and `s7` are both values greater than 1, it can be concluded that no broadcasting scenario exists for this node, thus determining that `s5` and `s7` are the same. Through the symbolic shape engine, the dynamic shape also has a certain shape judgment ability, the framework can complete more computational graph optimization functions based on this.

Detailed instructions for using the Symbol Engine can be found in the [Symbol API documentation](https://www.mindspore.cn/docs/en/r2.4.0/api_python/mindspore/mindspore.Symbol.html).

## Usage

MindSpore enables network dynamic shape compilation by setting the axis corresponding to the static graph input Tensor to dynamic via the set_inputs interface.
For example, to add two matrices, the size of the matrix changes. At this time we want to compile the computation logic corresponding to the matrix adding only once, and the same compilation process can be reused for calculating different sizes of matrices.
To set up dynamic shape compilation, you can use the symbol engine and the set_inputs interface to specify that the corresponding axis is dynamic, and the mindspore.jit decorator method can set it up using the input_signature parameter.
The following is an example of adding multiple different sizes of matrices:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Symbol

class Net(nn.Cell):
    def construct(self, x):
        return x + x

ms.context.set_context(mode=ms.context.GRAPH_MODE)
net = Net()
width = Symbol()
height = Symbol()
dyn_t = Tensor(shape=(width, height), dtype=ms.float32)
# Set Tensor shape dynamic
net.set_inputs(dyn_t)
# Execute with shape=(2 ,3)
input_x1 = Tensor(np.random.randn(2, 3), dtype=ms.float32)
out = net(input_x1)
# Execute with shape=(4, 5)
input_x2 = Tensor(np.random.randn(4, 5), dtype=ms.float32)
out = net(input_x2)
```

Detailed instructions for using set_inputs can be found in the [Cell.set_inputs API Ducumentation](https://www.mindspore.cn/docs/en/r2.4.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.set_inputs).

Detailed instructions for using input_signature can be found in the [mindspore.jit API Ducumentation](https://www.mindspore.cn/docs/en/r2.4.0/api_python/mindspore/mindspore.jit.html).

Distributed parallel scenarios on how to use dynamic shapes can be found in the [Distributed Parallel Support for Dynamic Shape Documentation](https://www.mindspore.cn/docs/en/r2.4.0/model_train/parallel/support_dynamic_shape_in_parallel.html).

## API Support

1. In the current version, only part of the API in MindSpore can support dynamic shape compilation and execution, and we will continue to improve the ability to support the full range of APIs. The current [mindspore.mint](https://www.mindspore.cn/docs/en/r2.4.0/api_python/mindspore.mint.html) interfaces support dynamic shape.
2. List[Tensor] and Tuple[Tensor] are not supported by set_inputs api for now.
