# 自定义算子

<a href="https://gitee.com/mindspore/docs/blob/r1.3/tutorials/source_zh_cn/intermediate/custom/operator.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## 自定义算子开发

MindSpore支持多种类型的算子，用户可根据[算子支持](https://www.mindspore.cn/docs/note/zh-CN/r1.3/operator_list.html)列表查询。若已有算子不满足实际需求，用户也可以开发自定义算子。

当前，MindSpore支持如下自定义算子，开发方法可通过链接获取：

- [自定义Ascend（昇腾）算子](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/custom_operator_ascend.html)
    - [AI Core算子](https://support.huaweicloud.com/tbedevg-cann330alpha2training/atlaste_10_0001.html)
    - [AI CPU算子](https://support.huaweicloud.com/aicpudevg-cann330alpha2training/atlasaicpu_10_0001.html)
- [自定义GPU算子](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/custom_operator_gpu.html)
- [自定义CPU算子](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/custom_operator_cpu.html)

在Ascend（昇腾）的两类算子中：

AI Core 算子是昇腾 AI 处理器计算核心的主要构成，负责执行向量和张量相关的计算密集型算子。TBE（Tensor Boost Engine）是一种在TVM（Tensor Virtual Machine）框架基础上扩展的算子开发工具，用户可使用 TBE 进行 AI Core 算子信息注册。

AI CPU算子是AI CPU负责执行昇腾处理器中海思 SoC 的CPU类算子（包括控制算子、标量和向量等通用计算）。MindSpore中同一个算子可能会同时拥有 AI Core 算子和AI CPU算子，框架会优先选择 AI Core 算子，没有 AI Core 算子或者不满足选择的场景下，会调用AI CPU算子。

在完成以上算子的开发之后，可参考下文在 MindSpore 中注册自定义算子。对于Ascend、GPU、CPU三种自定义算子，前端注册方式相同，进行同步说明。后端 Ascend 与GPU/CPU算子注册方式不同，将会分开说明。

## 自定义算子前端接入

下面以`mindspore.ops.Elu`算子为例，说明自定义算子前端接入 MindSpore 的具体方法。

Elu是一类激活函数，数学公式表达如下：

$$ ELU(x)=\begin{cases} \alpha(e^x-1) \qquad if \quad x \leq 0 \\\\ x \qquad\qquad\quad\ \  if \quad x > 0 \end{cases} $$

Elu 算子的输入为 Tensor，数据类型为 float16 或 float32 ，输出为同种数据类型、同种 shape 的 Tensor。当前系数$\alpha$仅支持设定为 float 类型的“1.0”。详细说明可查看[API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/ops/mindspore.ops.Elu.html)。

### 算子前端定义

#### 正向算子定义

Python 侧算子原语都定义在 MindSpore 仓的`mindspore/ops/operations/`文件夹中，根据算子的不同功能分类定义在不同的 Python 文件中。比如神经网络类算子定义在`nn_ops.py`，数学计算类的算子定义在`math_ops.py`等等。

Python 侧算子原语初始化过程中涉及很多属性值的校验，因此在`validator`类中定义了一批校验函数，具体位置在`mindspore/_checkparam.py`中。import以后可以直接调用。

算子原语定义可分为以下四步：

- 定义算子名：

    每个算子的原语是一个继承于 Primitive 的子类，类的名称即是算子名称，使用算子时即直接调用该算子原语。

- 确认算子输入的可写性（非必选）：

    根据算子情况，可使用`__mindspore_signature__`来校验输入是否满足要求。

- 注册算子属性：

    属性由构造函数`__init__`的入参定义，通过`prim_attr_register`装饰器将属性注册，入参的名字要和算子的属性名保持一致。

- 导入算子接口：

    注册完算子，需要在`mindspore/ops/operations/__init__.py`中对应的算子类别中添加算子名。导入注册好的算子原语，方便算子使用。

`nn_ops.py`中，Elu 算子的原语定义可以写成如下形式：

```Python
class Elu(PrimitiveWithInfer):
    """
    注册算子属性
    """
    @prim_attr_register
    def __init__(self, alpha=1.0):
        """
        初始化Elu，并进行校验
        检查公式中系数α是否为float形式，检查公式中系数α是否为1.0
        """
        validator.check_value_type("alpha", alpha, [float], self.name)
        validator.check_number("alpha", alpha, 1.0, Rel.EQ, self.name)

    def infer_shape(self, input_x):
        """shape推理函数"""
        return input_x

    def infer_dtype(self, input_x):
        """dtype推理函数，检查输入Tensor的类型是否有效"""
        validator.check_tensor_dtype_valid('input_x', input_x, mstype.float_type, self.name)
        return input_x
```

以`AssignSub`算子为例说明`__mindspore_signature__`的用法：

```python
class AssignSub(PrimitiveWithInfer):
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )
```

使用`sig.make_sig`进行校验，有几个输入就有几条`sig.make_sig`。

第一个入参是算子输入的名字，如果是要求Parameter类型，则增加`sig.sig_rw.RW_WRITE`入参，普通输入不传参，使用默认值即可。

第二个入参是进行计算的参数值，dtype是MindSpore中支持隐式转换的特性，使低精度类型向高精度类型进行转换。

#### 反向算子定义

根据算子是否要支持自动微分，用户可酌情添加反向算子。

- 算子属性注册

    与正向算子定义的步骤类似，首先在`mindspore/ops/operations/_grad_ops.py`文档中定义反向算子前端接口:

    ```python
    class EluGrad(PrimitiveWithInfer):

        @prim_attr_register
        def __init__(self):
            """初始化 EluGrad"""

        def infer_shape(self, y_grad_shape, x_shape):
            """shape推理函数，这里入参变更为所求梯度的shape与输入Tensor的shape"""
            return x_shape

        def infer_dtype(self, y_grad_dtype, x_dtype):
            """shape推理函数，这里入参变更为所求梯度的dtype与输入Tensor的dtype，并检查两者是否相同"""
            args = {'y_grad': y_grad_dtype, 'x': x_dtype}
            validator.check_tensors_dtypes_same_and_valid(args, mstype.float_type, self.name)
            return x_dtype

    ```

    反向算子没有特殊要求不对外公开，因此不需要在`mindspore/ops/operations/__init__.py`中添加反向算子名

- 反向算子注册

    MindSpore框架中的算子反向求导通过 Python 前端手工注册实现，代码地址为`mindspore/ops/_grad/`，每种算子类别都有对应的反向注册文件。

    例如，定义在`nn_ops.py`文件里的算子的反向放在对应的`grad_nn_ops.py`文件里。如果接入的算子有对应的反向算子，一般反向算子名就是正向算子名加上 Grad ， Elu 算子的反向算子就可以命名为 EluGrad ，根据反向逻辑实现。

    ```python
    import mindspore.ops as ops

    # 通过装饰器bprop_getters.register()注册
    @bprop_getters.register(ops.Elu)
    def get_bprop_elu(self):
        """定义Elu的反向，初始化反向算子，最后返回反向计算主体函数bprop"""
        input_grad = ops.EluGrad()

        def bprop(x, out, dout):
            """
            实现算子反向逻辑
            入参依次为算子正向输入Tensor、正向输出Tensor（统一命名为out）、和梯度（统一命名为dout），
            """
            dx = input_grad(dout, out)
            # 返回值包装为tuple
            return (dx,)

        return bprop
    ```

> 对`dout`进行说明：对于一个正向算子，它的梯度应该是这个算子本身对输入的梯度，而在反向传播时，要考虑到其他的计算。对于单个算子本身，如果其计算公式为$y = f(x)$，将使用该算子的其余计算简化为函数$l = g(y)$, 计算算子反向应该是 $dl/dx = dl/dy*dy/dx$，这里的$dl/dy$就是dout。
>
> 正向算子有几个输入就需要返回几个梯度值。如果只有一个输入，返回$dx$时需要包成一个tuple。如果有多个输入，但其中有几个无法计算梯度值时，使用`ops.ZerosLike()`算子返回和对应输入相同 shape 和 dtype ，值全为0的 Tensor 。以`MaxPool`的反向为例：

```python
@bprop_getters.register(ops.MaxPoolGrad)
def get_bprop_max_pool_grad_grad(self):

    def bprop(x1, x2, grad, out, dout):
        dx1 = zeros_like(x1)
        dx2 = zeros_like(x2)
        dgrad = maxpool_grad_grad(x1, x2, dout)
        return (dx1, dx2, dgrad)

    return bprop

```

## 自定义算子后端接入

### Ascend昇腾算子信息注册

正如前面所介绍，昇腾算子分为AI Core与AI CPU，两者以及相对应的反向算子信息注册方式相似，下面进行说明。

#### AI Core算子信息注册

在使用 TBE 完成AI Core算子开发后，Ascend 910 AI处理器配套软件包中会包含需要接入的算子信息，这些算子信息存放在 json 文件中。`/usr/local/Ascend`是配套软件包的安装路径，文件在环境中的地址为`/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json`。前端定义的算子接口应该与 json 文件中反映的算子相关信息是一致的。

仍然以Elu算子为例，算子信息如下：

```Shell
    "Elu":{
        "attr":{
            "list":"alpha"                 # 属性列表，算子底层计算是按 list 的属性顺序传入数据
        },
        "attr_alpha":{                     # 每个属性的相关信息
            "defaultValue":"1.0",
            "paramType":"optional",
            "type":"float",
            "value":"all"
        },
        "input0":{                         # 算子输入的相关信息
            "dtype":"float16,float",
            "name":"x",
            "paramType":"required"
        },
        "op":{                              # 算子数据格式说明
            "pattern":"formatAgnostic"
        },
        "output0":{                        # 算子输出的相关信息
            "dtype":"float16,float",
            "name":"y",
            "needCompile":"false",
            "paramType":"required",
            "shape":"all"
        },
        "slicePattern":{
            "value":"elemwise"
        }
    },

```

TBE 算子需要在`mindspore/ops/_op_impl/tbe`下根据 json 中的信息添加该算子对应的注册文件。文件名根据前端算子名转化而来。

Elu 算子注册文件为`elu.py`，具体注册内容如下：

```Python
from mindspore.ops import op_info_register, TBERegOp, DataType

elu_op_info = TBERegOp("Elu") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("elu.so") \
    .compute_cost(10) \
    .kernel_name("elu") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .attr("alpha", "optional", "float", "all", "1.0") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(elu_op_info)
def _elu_tbe():
    """Elu TBE算子注册"""
    return

```

注册完后要在`mindspore/ops/_op_impl/tbe/__init__.py`中导入注册好的算子信息。

```python
from mindspore.ops._op_impl.tbe import _elu_tbe
```

#### AI CPU算子信息注册

AI CPU 算子的注册文件放在`mindspore/ops/_op_impl/aicpu/`文件夹中，注册方式与AI Core 算子类似。注册完后要在`mindspore/ops/_op_impl/aicpu/__init__.py`中导入注册好的算子信息。

#### 反向算子信息注册

反向算子信息注册的方式可参考上文中正向算子的相关内容。
Elu 算子反向信息注册在`mindspore/ops/_op_impl/tbe/elu_grad.py`文件中。注册完成之后同样需要在`mindspore/ops/_op_impl/tbe/__init__.py`中导入算子信息。

### GPU/CPU算子信息注册

同样以 Elu 的 GPU 、CPU 算子为例进行说明。

#### 正向算子信息注册

算子信息包含:

- `Primive`：算子名称
- `Input dtype, output dtype`：输入输出的数据格式
- `GPU Kernel class`或`CPU Kernel class`：GPU或 CPU 后端
- `内置数据类型`

GPU、CPU后端注册位于`mindspore/ccsrc/backend/kernel_compiler/`目录下，`kernel_compiler/gpu/nn/activation_gpu_kernel.cc`通过四类信息实现了 Elu GPU 正向算子注册：

```cpp
namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Elu, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ActivationGpuFwdKernel, float)
MS_REG_GPU_KERNEL_ONE(Elu, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ActivationGpuFwdKernel, half)
}  // namespace kernel
}  // namespace mindspore

```

在头文件`activation_gpu_kernel.h`中完成初始化与校验。

在`kernel_compiler/cpu/elu_grad_cpu_kernel.cc`，`kernel_compiler/cpu/elu_grad_cpu_kernel.h`中完成了Elu CPU正向算子注册、初始化与校验，实现方式与 GPU 类似。

#### 反向算子信息注册

> 根据算子是否要支持自动微分，用户可酌情添加反向算子。

与正向算子类似，在`mindspore/ccsrc/backend/kernel_compiler/gpu/nn/activation_grad_kernel.cc`中分别注册了支持多种数据类型的EluGrad GPU算子：

```cpp
namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  EluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ActivationGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  EluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ActivationGradGpuKernel, half)
}  // namespace kernel
}  // namespace mindspore

```

在头文件`activation_grad_kernel.h`中完成初始化与校验。
