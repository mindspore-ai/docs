# 自定义算子（Ascend）

`Ascend` `模型开发`

<!-- TOC -->

- [自定义算子（Ascend）](#自定义算子ascend)
    - [概述](#概述)
    - [注册算子原语](#注册算子原语)
    - [实现TBE算子和注册算子信息](#实现tbe算子和注册算子信息)
        - [实现TBE算子](#实现tbe算子)
        - [注册算子信息](#注册算子信息)
        - [示例](#示例)
     - [实现AICPU算子和注册算子信息](#实现AICPU算子和注册算子信息)
       - [实现AICPU算子](#实现AICPU算子)
       - [注册AICPU自定义算子信息](#注册AICPU自定义算子信息)
       - [示例](#示例)
    - [使用自定义算子](#使用自定义算子)
    - [定义算子反向传播函数](#定义算子反向传播函数)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/custom_operator_ascend.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API方便快捷地扩展昇腾AI处理器的自定义算子。

添加一个自定义算子，需要完成算子原语注册、算子实现、算子信息注册三部分工作。

其中：  

- 算子原语：定义了算子在网络中的前端接口原型，也是组成网络模型的基础单元，主要包括算子的名称、属性（可选）、输入输出名称、输出shape推理方法、输出dtype推理方法等信息。
- 算子实现：通过TBE（Tensor Boost Engine）提供的特性语言接口，描述算子内部计算逻辑的实现。TBE提供了开发昇腾AI芯片自定义算子的能力。
- 算子信息：描述TBE算子的基本信息，如算子名称、支持的输入输出类型等。它是后端做算子选择和映射时的依据。

本文将以自定义Square算子为例，介绍自定义算子的步骤。

> 更多详细内容可参考MindSpore源码中[tests/st/ops/custom_ops_tbe](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/custom_ops_tbe)下的用例。

## 注册算子原语

每个算子的原语是一个继承于`PrimitiveWithInfer`的子类，其类型名称即是算子名称。

自定义算子原语与内置算子原语的接口定义完全一致：  

- 属性由构造函数`__init__`的入参定义。本用例的算子没有属性，因此`__init__`没有额外的入参。带属性的用例可参考MindSpore源码中的[custom add3](https://gitee.com/mindspore/mindspore/blob/master/tests/st/ops/custom_ops_tbe/cus_add3.py)用例。
- 输入输出的名称通过`init_prim_io_names`函数定义。
- 输出Tensor的shape推理方法在`infer_shape`函数中定义，输出Tensor的dtype推理方法在`infer_dtype`函数中定义。

自定义算子与内置算子的唯一区别是需要通过在`__init__`函数中导入算子实现函数(`from square_impl import CusSquareImpl`)来将算子实现注册到后端。本用例在`square_impl.py`中定义了算子实现和算子信息，将在后文中说明。

以Square算子原语`cus_square.py`为例，给出如下示例代码。

```python
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
import mindspore.ops as ops
# y = x^2
class CusSquare(PrimitiveWithInfer):
    """
    The definition of the CusSquare primitive.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from square_impl import CusSquareImpl # Import the entry function of the kernel implementation from relative path or PYTHONPATH.

    def infer_shape(self, data_shape):
        return data_shape

    def infer_dtype(self, data_dtype):
        return data_dtype
```

## 实现TBE算子和注册算子信息

### 实现TBE算子

通常编写一个算子的实现，需要编写一个计算函数和一个入口函数。

算子的计算函数主要用来封装算子的计算逻辑供主函数调用，其内部通过调用TBE的API接口组合实现算子的计算逻辑。

算子的入口函数描述了编译算子的内部过程，一般分为如下几步：  

1. 准备输入的placeholder，placeholder是一个占位符，返回一个Tensor对象，表示一组输入数据。
2. 调用计算函数，计算函数使用TBE提供的API接口描述了算子内部的计算逻辑。
3. 调用Schedule调度模块，调度模块对算子中的数据按照调度模块的调度描述进行切分，同时指定好数据的搬运流程，确保在硬件上的执行达到最优。默认可以采用自动调度模块（`auto_schedule`）。
4. 调用`cce_build_code`编译生成算子二进制。

> 入口函数的输入参数有特殊要求，需要依次为：算子每个输入的信息、算子每个输出的信息、算子属性（可选）和`kernel_name`（生成算子二进制的名称）。输入和输出的信息用字典封装传入，其中包含该算子在网络中被调用时传入的实际输入和输出的shape和dtype。

更多关于使用TBE开发算子的内容请参考[TBE文档](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0063.html)，关于TBE算子的调试和性能优化请参考[MindStudio文档](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0043.html)。

### 注册算子信息

算子信息是指导后端选择算子实现的关键信息，同时也指导后端为算子插入合适的类型和格式转换。它通过`TBERegOp`接口定义，通过`op_info_register`装饰器将算子信息与算子实现入口函数绑定。当算子实现py文件被导入时，`op_info_register`装饰器会将算子信息注册到后端的算子信息库中。更多关于算子信息的使用方法请参考`TBERegOp`的成员方法的注释说明，算子信息的字段含义可以参考[TBE文档](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0096.html)。

> - 算子信息中定义输入输出信息的个数和顺序、算子实现入口函数的参数中的输入输出信息的个数和顺序、算子原语中输入输出名称列表的个数和顺序，三者要完全一致。
> - 算子如果带属性，在算子信息中需要用`attr`描述属性信息，属性的名称与算子原语定义中的属性名称要一致。

### 示例

下面以`Square`算子的TBE实现`square_impl.py`为例进行介绍。`square_compute`是算子实现的计算函数，通过调用`te.lang.cce`提供的API描述了`x * x`的计算逻辑。`cus_square_op_info`是算子信息，通过`TBERegOp`来定义。

`TBERegOp`的设置需要注意以下几点：

- `TBERegOp("CusSquare")`中算子注册名称`CusSquare`需要与算子名称一致。
- `fusion_type("OPAQUE")`中`OPAQUE`表示自定义算子采取不融合策略。
- `kernel_name("CusSquareImpl")`中`CusSquareImpl`需要与算子入口函数名称一致。
- `dtype_format`用来描述算子支持的数据类型，下面示例中注册了两项，说明该算子支持两种数据类型，每一项需按照输入和输出的顺序依次描述支持的格式。第一个`dtype_format`说明支持的第一种数据类型是input0为F32_Default格式，output0为F32_Default格式。第二个`dtype_format`说明支持的第二种数据类型是input0为F16_Default格式，output0为F16_Default格式。
- `auto_schedule`、`cce_build_code`等TBE相关接口描述请见TBE文档中[auto_schedule](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_07_0071.html)和[cce_build_code](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_07_0072.html)的详细说明。

```python
from __future__ import absolute_import
from te import tvm
from topi import generic
import te.lang.cce
from topi.cce import util
from mindspore.ops import op_info_register, TBERegOp, DataType

def square_compute(input_x):
    """
    The compute function of the CusSquare implementation.
    """
    res = te.lang.cce.vmul(input_x, input_x)
    return res

# Define the kernel info of CusSquare.
cus_square_op_info = TBERegOp("CusSquare") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("square.so") \
    .compute_cost(10) \
    .kernel_name("CusSquareImpl") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()

# Binding kernel info with the kernel implementation.
@op_info_register(cus_square_op_info)
def CusSquareImpl(input_x, output_y, kernel_name="CusSquareImpl"):
    """
    The entry function of the CusSquare implementation.
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res = square_compute(data)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
```

自定义算子与内置算子在网络中的使用方法一样，通过导入原语直接使用。下面以`CusSquare`的单算子网络测试为例进行说明。

在`test_square.py`文件中定义网络。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
# Import the definition of the CusSquare primitive.
from cus_square import CusSquare
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.square = CusSquare()

    def construct(self, data):
        return self.square(data)

def test_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    square = Net()
    output = square(Tensor(x))
    print("x: ", x)
    print("output: ", output)
```

执行用例:

```bash
pytest -s tests/st/ops/custom_ops_tbe/test_square.py::test_net
```

执行结果:

```text
x: [1. 4. 9.]
output: [1. 16. 81.]
```

## 实现AICPU算子和注册算子信息

### 实现AICPU算子

基于CANN开发AICPU算子包含算子原型定义、算子代码实现、算子信息库定义等步骤，具体开发步骤请参考[CANN AICPU 自定义算子开发](https://support.huaweicloud.com/usermanual-mindstudio303/atlasms_02_0194.html)。

开发完成之后将编译生成一个指定名称的文件，如`libmindspore_aicpu_kernels.so`，`libcust_reshape.so`这类文件，这些动态库中可包含一个或多个AICPU算子实现，将该文件放到MindSpore安装或者编译目录下的lib目录下，MindSpore即可通过后续自定义算子注册信息加载该文件。

> 算子实现的动态库文件，需要放到MindSpore的lib目录下，比如MindSpore安装在虚拟环境`/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore`下，则aicpu的so文件需要放到`/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore/lib/`目录下，这样即可正常加载到文件。

更多关于AICPU算子的调试和性能优化请参考[MindStudio文档](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0043.html)。

### 注册AICPU自定义算子信息

在完成上一步后，跟TBE算子一致，我们需要补充算子信息。AICPU算子通过`AiCPURegOp`接口定义，通过`op_info_register`装饰器将算子信息与算子实现入口函数绑定。当算子实现py文件被导入时，`op_info_register`装饰器会将算子信息注册到后端的算子信息库中。更多关于算子信息的使用方法请参考`AiCPURegOp`的成员方法的注释说明，算子信息的字段含义可以参考[AICPU文档](https://support.huaweicloud.com/usermanual-mindstudio303/atlasms_02_0194.html)。

> - 算子信息中定义输入输出信息的个数和顺序、算子实现入口函数的参数中的输入输出信息的个数和顺序、算子原语中输入输出名称列表的个数和顺序，三者要完全一致。
> - 算子如果带属性，在算子信息中需要用`attr`描述属性信息，属性的名称与算子原语定义中的属性名称要一致。

需要额外注意的是，在基础的注册信息外，我们需要额外添加`attr("cust_aicpu", "str")`属性，该属性是用于获取算子实现的so名称。以`RandomChoiceWithMask`算子为例，假设我们已经定义好了算子原语，并且算子实现已经编译为`librandom_choice_with_mask.so`，那么我们只需要在算子信息库中添加`attr("cust_aicpu", "str")`，然后在算子定义时，设置该属性值为`"random_choice_with_mask"`即可完成将该算子注册到自定义AICPU算子列表中。

> “cust_aicpu”的值为字符串，用算子so的名字去除`lib`前缀与`.so`后缀表示，如`libmindspore_aicpu_kernels.so`则设为`"mindspore_aicpu_kernels"`即可。

```python
from mindspore.ops import op_info_register, AiCPURegOp, DataType

random_choice_with_mask_op_info = AiCPURegOp("RandomChoiceWithMask") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .output(1, "mask", "required") \
    .attr("count", "int") \
    .attr("seed", "int") \
    .attr("seed2", "int") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.BOOL_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .get_op_info()

@op_info_register(random_choice_with_mask_op_info)
def _random_choice_with_mask_aicpu():
    """RandomChoiceWithMask AiCPU register"""
    return
```

### 示例

下面以`Dropout2D`算子的AICPU调用实现为例进行介绍，我们会经历算子实现、算子原语注册、算子信息库、算子调用四个步骤：

1. 算子实现：参考[实现AICPU算子](#实现AICPU算子)的相关内容，我们将算子编译成`libmindspore_aicpu_kernels.so`。
2. 算子原语注册：参考[注册算子原语](#注册算子原语)的相关内容，我们将定义一个Dropout2D的算子。
3. 算子信息库：参考[注册AICPU自定义算子信息](#注册AICPU自定义算子信息)的相关内容，我们将实现Dropout2D的信息库，并且添加`"cust_aicpu"`的属性。
4. 算子调用：我们可以正常按照单算子网络的形式调用Dropout2D算子，同时可以配置`"cust_aicpu"`的属性值为`mindspore_aicpu_kernels`。

```python
import numpy as np
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore import dtype as mstype
from mindspore.ops import op_info_register, AiCPURegOp, DataType
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Dropout2D(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, keep_prob=0.5):
        """Initialize Dropout2D."""
        pass

    def infer_shape(self, x_shape):
        return x_shape, x_shape

    def infer_dtype(self, x_dtype):
        mask_dtype = mstype.tensor_type(mstype.bool_)
        return x_dtype, mask_dtype

dropout2d_op_info = AiCPURegOp("Dropout2D") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .output(1, "mask", "required") \
    .attr("keep_prob", "float") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.BOOL_Default) \
    .get_op_info()

@op_info_register(dropout2d_op_info)
def _dropout2d_aicpu():
    """Dropout2D AiCPU register"""
    return

class NetDropout2D(nn.Cell):
    def __init__(self, keep_prob=0.5):
        super(NetDropout2D, self).__init__()
        self.op = Dropout2D(keep_prob)
        self.op.add_prim_attr("cust_aicpu", "mindspore_aicpu_kernels")

    def construct(self, inputs):
        return self.op(inputs)

if __name__ == "__main__":
    input_tensor = Tensor(np.ones([1, 1, 2, 3]), mstype.float32)
    dropout2d_nn = NetDropout2D(0.5)
    output, mask = dropout2d_nn(input_tensor)
    print("output: ", output)
    print("mask: ", mask)
```

执行结果:

```text
output: [[[[0.0.0.]
  [0.0.0.]]]]
mask: [[[[False False False]
  [False False False]]]]
```

## 定义算子反向传播函数

如果算子要支持自动微分，需要在其原语中定义其反向传播函数（bprop）。你需要在bprop中描述利用正向输入、正向输出和输出梯度得到输入梯度的反向计算逻辑。反向计算逻辑可以使用内置算子或自定义反向算子构成。

定义算子反向传播函数时需注意以下几点：

- bprop函数的入参顺序约定为正向的输入、正向的输出、输出梯度。若算子为多输出算子，正向输出和输出梯度将以元组的形式提供。
- bprop函数的返回值形式约定为输入梯度组成的元组，元组中元素的顺序与正向输入参数顺序一致。即使只有一个输入梯度，返回值也要求是元组的形式。

例如，增加bprop后的`CusSquare`原语为：

```python
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
import mindspore.ops as ops

class CusSquare(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init CusSquare"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from square_impl import CusSquareImpl

    def infer_shape(self, data_shape):
        return data_shape

    def infer_dtype(self, data_dtype):
        return data_dtype

    def get_bprop(self):
        def bprop(data, out, dout):
            twos_like = ops.OnesLike()(data) * 2.0
            gradient = ops.Mul()(data, twos_like)
            dx = ops.Mul()(gradient, dout)
            return (dx,)
        return bprop
```

在`test_square.py`文件中定义反向用例。

```python
import mindspore.ops as ops
def test_grad_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    square = Net()
    grad = ops.GradOperation(sens_param=True)
    dx = grad(square)(Tensor(x), Tensor(sens))
    print("x: ", x)
    print("dx: ", dx)
```

执行用例:

```bash
pytest -s tests/st/ops/custom_ops_tbe/test_square.py::test_grad_net
```

执行结果:

```text
x: [1. 4. 9.]
dx: [2. 8. 18.]
```
