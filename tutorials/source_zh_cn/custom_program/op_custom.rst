自定义算子
============

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_notebook.svg
    :target: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/br_base/tutorials/zh_cn/custom_program/operation/mindspore_op_custom.ipynb
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_download_code.svg
    :target: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/br_base/tutorials/zh_cn/custom_program/operation/mindspore_op_custom.py
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/custom_program/operation/op_custom.ipynb
    :alt: 查看源文件

.. toctree::
   :maxdepth: 1
   :hidden:

   operation/op_custom_ascendc
   operation/op_custom_aot
   operation/op_custom_julia
   operation/op_custom_adv
   operation/op_custom_pyboost

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API中的 `Custom <https://www.mindspore.cn/docs/zh-CN/br_base/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom>`_ 原语方便快捷地进行不同类型自定义算子的定义和使用。

传统的添加一个自定义算子的方式，需要完成算子原语注册、算子实现、算子信息注册三部分工作。

其中：

- 算子原语：定义了算子在网络中的前端接口原型，也是组成网络模型的基础单元，主要包括算子的名称、属性（可选）、输入输出名称、输出shape推理方法、输出数据类型推理方法等信息。
- 算子实现：在Python侧定义函数（JIT类型自定义算子）或C++侧定义类（GPU和CPU自定义算子），描述算子内部计算逻辑的实现。
- 算子信息：描述自定义算子的基本信息，如算子名称、支持的输入输出数据类型、支持的输入输出数据格式和属性等。它是后端做算子选择和映射时的依据。

相比于传统自定义算子方式，基于 `Custom` 原语自定义算子具有如下优势：

- 不同的自定义算子对应的算子原语都是 `Custom` 原语，无需对每个自定义算子定义一个相应的算子原语。上述提到的三部分工作可以在网络脚本中以统一的接口进行实现，并作为网络表达的一部分，不需要对MindSpore框架进行侵入式修改和重新编译。
- 实现了不同方式自定义算子的接口和使用统一，方便网络开发者根据需要灵活选用不同的自定义方式。

自定义算子分类及适应场景
------------------------

基于 `Custom <https://www.mindspore.cn/docs/zh-CN/br_base/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom>`_ 原语的自定义算子支持的算子开发方式包括：pyfunc、aot和julia。不同的算子开发方式适应的场景如下：

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - 算子开发方式
     - 开发语言
     - 支持平台
     - 推荐场景
   * - `pyfunc <#自定义算子用例>`_
     - Python
     - `CPU`
     - 快速算法验证的场景
   * - `aot <https://www.mindspore.cn/tutorials/zh-CN/br_base/custom_program/operation/op_custom_aot.html>`_
     - Ascend C/CUDA/C++
     - `Ascend` `GPU` `CPU`
     - 需要高性能算子的场景
   * - `julia <https://www.mindspore.cn/tutorials/zh-CN/br_base/custom_program/operation/op_custom_julia.html>`_
     - Julia
     - `CPU`
     - 科学计算场景

不同的开发方式使用不同的开发语言实现算子计算逻辑，但是自定义算子的开发流程是一致的，包括算子实现、shape推导、数据类型推理和算子信息注册（可选）。网络开发者可以根据需要选用不同的自定义算子开发方式。在开发者进行自定义算子开发的时候，可以参考如下方式选择对应类型：

1. 判断后端：如果用户使用的是Ascend和GPU后端，那么就选择aot类型的自定义算子；如果是CPU后端，则根据使用的场景选择；

2. 判断场景：在使用CPU后端的时候，不同的场景对应不同类型的自定义算子推荐：
   
   - 快速验证场景：如果用户希望基于MindSpore做快速验证和开发，对于性能要求不高，或者希望基于Python进行交互，那么选取pyfunc类型的自定义算子；
   - 高性能场景：如果用户希望基于MindSpore做高性能计算，或者需要对接第三方算子库，那么选取aot类型自定义算子；
   - 科学计算场景：如果用户在做科学计算任务时需要使用Julia，那么选取julia类型自定义算子。

为了帮助大家更好地使用自定义算子，我们以 `pyfunc类型自定义算子 <#自定义算子用例>`_ 中作为自定义算子的范例展示。此外，我们提供了其他自定义算子的教程包括：

- aot类型自定义算子： `Ascend平台 <https://www.mindspore.cn/tutorials/zh-CN/br_base/custom_program/operation/op_custom_ascendc.html>`_ 和 `GPU/CPU平台 <https://www.mindspore.cn/tutorials/zh-CN/br_base/custom_program/operation/op_custom_aot.html>`_ ；
- `julia类型自定义算子 <https://www.mindspore.cn/tutorials/zh-CN/br_base/custom_program/operation/op_custom_julia.html>`_ ；
- `自定义算子进阶用法 <https://www.mindspore.cn/tutorials/zh-CN/br_base/custom_program/operation/op_custom_adv.html>`_ ：算子注册和反向算子。

.. note::
   更多示例可参考MindSpore源码中 `tests/st/graph_kernel/custom <https://gitee.com/mindspore/mindspore/tree/br_base/tests/st/graph_kernel/custom>`_ 下的用例。

自定义算子用例
--------------

为了帮助用户快速入门自定义算子，这里以pyfunc类型自定义算子为例帮助用户理解自定义算子的定义流程。下面基于pyfunc模式定义一个实现sin计算的自定义算子。pyfunc类型的自定义算子使用原生Python语法定义算子实现函数，描述算子内部计算逻辑的实现。网络运行时框架会自动调用此函数。为了表达自定义算子的计算，我们写一个基于numpy的计算正弦函数的Python原生函数。

.. code-block:: python

    import numpy as np

    def sin_by_numpy(x):
        return np.sin(x)

然后我们要定义两个函数，一个是张量形状的推导函数（infer_shape），另一个是张量数据类型的推导函数（infer_dtype）。这里要注意：

- 张量形状的推导函数是输入张量的形状；
- 张量数据类型的推导函数是输入张量的数据类型。

.. code-block:: python

    def infer_shape(x):
        #    1. 这里的输入x是算子输入张量的形状
        #    2. sin函数是逐元素计算，输入的形状和输出的一样
        return x

    def infer_dtype(x):
        #    1. 这里的输入x是算子输入张量的数据类型
        #    2. sin函数输入的数据类型和输出的一样
        return x

下面我们用上面的函数自定义一个算子，其输入包括

- func：自定义算子的函数表达，这里我们用 `sin_by_numpy` 函数；
- out_shape: 输出形状的推导函数，这里我们用 `infer_shape` 函数；
- out_dtype: 输出数据类型的推导函数，这里我们用 `infer_dtype` 函数；
- func_type: 自定义算子类型，这里我们用 `pyfunc`。

.. code-block:: python
    
    from mindspore import ops

    sin_by_numpy_op = ops.Custom(func=sin_by_numpy, # 这里填入自定义算子的函数表达
                                 out_shape=infer_shape, # 这里填入输出形状的推导函数
                                 out_dtype=infer_dtype, # 这里填入输出数据类型的推导函数
                                 func_type="pyfunc" # 这里填入自定义算子类型
                                 )

加上其他环境依赖依赖和算子调用语句，我们获得完整的自定义算子用例如下。

.. code-block:: python

    import numpy as np
    import mindspore as ms
    from mindspore import ops

    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device(device_target="CPU")

    def sin_by_numpy(x):
        return np.sin(x)

    def infer_shape(x):
        return x

    def infer_dtype(x):
        return x

    sin_by_numpy_op = ops.Custom(func=sin_by_numpy,
                                 out_shape=infer_shape,
                                 out_dtype=infer_dtype,
                                 func_type="pyfunc")
   
    input_tensor = ms.Tensor([0, 1, 0.2, 0.3, 0.4], dtype=ms.float32)
    result_cus = sin_by_numpy_op(input_tensor)
    print(result_cus)

我们可以得到结果即为上面输入对应的sin值。

.. raw:: html

    <div class="highlight"><pre>
    [0.         0.841471   0.19866933 0.29552022 0.38941833]
    </pre></div>

如此我们完成一个pyfunc类型自定义算子的定义。对于更多完整的pyfunc类型自定义算子的例子，参见MindSpore源码中的 `用例 <https://gitee.com/mindspore/mindspore/blob/br_base/tests/st/graph_kernel/custom/test_custom_pyfunc.py>`_ 。
