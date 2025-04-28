Custom Primitive-Based Custom Operators
========================================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/custom_program/operation/op_custom_prim.rst
    :alt: View Source On Gitee

When built-in operators cannot meet requirements during network development, you can call the Python API `Custom <https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom>`_ primitive defined in MindSpore to quickly create different types of custom operators for use.

Traditional methods to add a custom operator need three steps: registering the operator primitive, implementing the operator, and registering the operator information.

The related concepts are as follows:

- Operator primitive: defines the frontend API prototype of an operator on the network. It is the basic unit for forming a network model and includes the operator name, attribute (optional), input and output names, output shape inference method, and output data type inference method.
- Operator implementation: defines a Python function (JIT custom operators) or a C++ class (GPU and CPU custom operators), which describes the implementation of the internal computation logic of an operator.
- Operator information: describes basic information about an operator, such as the operator name, supported input and output data types, supported input and output data formats, and attributes. It is the basis for the backend to select and map operators.

Compared with traditional custom operator creating methods, creating custom operators based on `Custom` primitive has several advantages:

- Different custom operators use the same `Custom` primitive, there is no need to define a primitive for every operator. The above three parts of work can be implemented in a network script in a unified way and used as part of the network expression, there is no need to modify and recompile the source codes of MindSpore.
- It unifies the interface and usage for different kinds of custom operators, which is convenient for network developers to flexibly choose which kind of custom operator to use according to their needs.

Custom operator classification and adaptation scenarios
-----------------------------------------------------------

The operator development methods supported by custom operator based on the `Custom <https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom>`_ primitive include: aot, pyfunc, and julia.

The difference between these operator development methods are as follows:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Defining Methods
     - Development Language
     - Supported Platforms
     - Recommended Scenarios
   * - `pyfunc <#an-example-of-custom-operators>`_
     - Python
     - `CPU`
     - Fast algorithm verification scenarios
   * - `aot <https://www.mindspore.cn/tutorials/en/r2.6.0/custom_program/operation/op_custom_aot.html>`_
     - Ascend C/CUDA/C++
     - `Ascend` `GPU` `CPU`
     - high-performance scenarios
   * - `julia <https://www.mindspore.cn/tutorials/en/r2.6.0/custom_program/operation/op_custom_julia.html>`_
     - Julia
     - `CPU`
     - Science compute scenarios / use Julia scenarios

Different custom operator defining methods use different development languages to implement the operator, but the development process is the same, including operator implementation, shape inference, data type inference, and operator information registration (optional). Developers can choose which one to use based on needs. The defining methods of these custom operators will be introduced here, and examples are provided for each method. When developers are developing custom operators, they can refer to the following methods to select the corresponding type:

1. Identify the backend: If the user is using the Ascend and GPU backend, then choose the AOT type of custom operator; If it is a CPU backend, choose according to the usage scenario;
2. Identify the scenario: When using the CPU backend, different scenarios correspond to different types of custom operators. Recommendations:

   - Quick verification scenario: If users want to do quick verification and development based on MindSpore, with low performance requirements, or want to interact based on Python, then choose custom operators with Pyfunc type;
   - High-performance scenario: If users want to do high-performance computing based on MindSpore or need to interface with third-party operator libraries, then choose custom operators of AOT type;
   - Scientific computing scenario: If users need to use Julia for scientific computing tasks, then choose custom operators of Julia type.

To help you better use custom operators, we have used [the pyfunc-type custom operator](#an-example-of-custom-operators) as an example of a custom operator. In addition, we provide tutorials for other custom operators including:

- AOT-type custom op on `Ascend backend <https://www.mindspore.cn/tutorials/en/r2.6.0/custom_program/operation/op_custom_ascendc.html>`_ and `GPU/CPU backend <https://www.mindspore.cn/tutorials/en/r2.6.0/custom_program/operation/op_custom_aot.html>`_ ;
- `Julia-type custom op <https://www.mindspore.cn/tutorials/en/r2.6.0/custom_program/operation/op_custom_julia.html>`_ ;
- `Advanced usage of custom operators <https://www.mindspore.cn/tutorials/en/r2.6.0/custom_program/operation/op_custom_adv.html>`_ : registering the operator information and defining the backward functions for operators.

.. note::
    More examples can be found in the MindSpore `source code <https://gitee.com/mindspore/mindspore/tree/v2.6.0/tests/st/graph_kernel/custom>`_ .

An Example of Custom Operators
--------------------------------

To help users quickly get started with custom operators, here is an example of a pyfunc type custom operator to help users understand the definition process of custom operators.
The following defines a custom operator that implements the sin calculation based on the pyfunc pattern.
Custom operators of type pyfunc use native Python syntax to define operator implementation functions, describing the implementation of the operator's internal computational logic.
During the network runtime, framework will automatically call this function. In order to express the calculation of a custom operator, we write a Python native function based on numpy to calculate the sine function.

.. code-block:: python

    import numpy as np

    def sin_by_numpy(x):
        return np.sin(x)

Then we need to define two more functions. One is the infer shape function, while the other is the infer dtype function. Here's something to keep in mind:

- The derivative function of the tensor shape is the shape of the input tensor;
- The derivative function of the tensor dtype is the dtype of the input tensor;

.. code-block:: python

    def infer_shape(x):
        #    1. here x is the shape of the input tensor
        #    2. sin is elements, so the shape of the output is the same as that of the input.
        return x

    def infer_dtype(x):
        #    1. here x is the dtype of the input tensor
        #    2. sin keeps the dtype, so the dtype of the output is the same as that of the input.
        return x

Then we use the above functions to create a custom operator, and the inputs include:

- func: the computation function of the custom op. Here we use `sin_by_numpy` above;
- out_shape: the infer shape function. Here we use `infer_shape` above;
- out_dtype: the infer dtype function. Here we use `infer_dtype` above;
- func_type: the mode of the custom operator. Here we use `"pyfunc"`.

.. code-block:: python
    
    from mindspore import ops

    sin_by_numpy_op = ops.Custom(func=sin_by_numpy, # this is for the computation function
                                 out_shape=infer_shape, # this is for the infer shape function
                                 out_dtype=infer_dtype, # this is for the infer dtype function
                                 func_type="pyfunc" # this is for the custom op mode
                                 )

Adding other environment dependencies and operator call statements, we obtain the complete custom operator use case as follows.

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

Then we have the following results as sin values of above inputs.

.. raw:: html

    <div class="highlight"><pre>
    [0.         0.841471   0.19866933 0.29552022 0.38941833]
    </pre></div>

Then we have completed the definition of a custom operator of type pyfunc. For more complete examples of pyfunc-type custom operators, see the `use cases <https://gitee.com/mindspore/mindspore/blob/v2.6.0/tests/st/graph_kernel/custom/test_custom_pyfunc.py>`_ in the MindSpore source code.
