# Constraints on Network Construction Using Python

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.6/docs/source_en/constraints_on_network_construction.md)

## Overview
  MindSpore can compile user source code based on the Python syntax into computational graphs, and can convert common functions or instances inherited from nn.Cell into computational graphs. Currently, MindSpore does not support conversion of any Python source code into computational graphs. Therefore, there are constraints on source code compilation, including syntax constraints and network definition constraints. As MindSpore evolves, the constraints may change.

## Syntax Constraints
### Supported Python Data Types
* Number: supports `int`, `float`, and `bool`. Complex numbers are not supported.
* String
* List: supports the append method only. Updating a list will generate a new list.
* Tuple
* Dictionary: The type of key should be String.
### MindSpore Extended Data Type
* Tensor: Tensor variables must be defined instances.

### Expression Types

| Operation               | Description 
| :-----------            |:--------
| Unary operator          |`+`,`-`, and`not`. The operator `+` supports only scalars.
| Binary operator         |`+`, `-`, `*`, `/`, `%`, `**` and `//`.
| `if` expression         | For example, `a = x if x < y else y`.
| Comparison expression   | `>`, `>=`, `<`, `<=`, `==`, and `! =`.
| Logical expression      | `and` and `or`.
| `lambda` expression     | For example, `lambda x, y: x + y`.
| Reserved keyword type   | `True`, `False`, and `None`.

### Statement Types

| Statement    | Compared with Python
| :----------- |:--------
| `def`        | Same as that in Python.
| `for`        | Nested for loops are partially supported. Iteration sequences must be tuples or lists.
| `while`      | Nested while loops are partially supported. Grad of net with while is not supported.
| `break`      | Same as that in Python.
| `if`         | Same as that in Python. The input of the `if` condition must be a constant.
| `in`         | Only supports judging whether constants exist in Tuple/List/Dictionary whose elements are all constants.
| `not in`     | Only support Dictionary.
| Assignment statement     | Accessed multiple subscripts of lists and dictionaries cannot be used as l-value.

### System Functions/Classes

| Functions/Class         | Compared with Python
| :----------- |:--------
| `len`        | The usage principle is consistent with Python, and the returned result is consistent with Python, returning int.
| `partial`    | The usage principle is consistent with Python, and the returned result is inconsistent with Python, returning function.
| `map`        | The usage principle is consistent with Python, and the returned result is inconsistent with Python, returning tuple.
| `zip`        | The usage principle is consistent with Python, and the returned result is inconsistent with Python, returning tuple.
| `range`      | The usage principle is consistent with Python, and the returned result is inconsistent with Python, returning tuple.
| `enumerate`  | The usage principle is consistent with Python, and the returned result is inconsistent with Python, returning tuple.
| `super`      | The usage principle is consistent with Python, and the returned result is inconsistent with Python, returning the namespace defined by mindspore.
| `isinstance` | The usage principle is consistent with Python, but the second input parameter can only be the type defined by mindspore.

### Function Parameters
*  Default parameter value: The data types `int`, `float`, `bool`, `None`, `str`, `tuple`, `list`, and `dict` are supported, whereas `Tensor` is not supported.
*  Variable parameter: Functions with variable arguments is supported for training and inference.
*  Key-value pair parameter: Functions with key-value pair parameters cannot be used for backward propagation on computational graphs.
*  Variable key-value pair parameter: Functions with variable key-value pairs cannot be used for backward propagation on computational graphs.

### Operators

| Operator     | Supported Type
| :----------- |:--------
| `+`          |Scalar, `Tensor`, `tuple` and `string`
| `-`          |Scalar and `Tensor`
| `*`          |Scalar and `Tensor`
| `/`          |Scalar and `Tensor`
| `**`         |Scalar and `Tensor`
| `//`         |Scalar and `Tensor`
| `%`          |Scalar and `Tensor`
| `[]`         |The operation object type can be `list`, `tuple`, or `Tensor`. Accessed multiple subscripts of lists and dictionaries can be used as r-values instead of l-values. The index type cannot be Tensor. For details about access constraints for the tuple and Tensor types, see the description of slicing operations.

### Index operation

The index operation includes `tuple` and` Tensor`. The following focuses on the index value assignment and assignment operation of `Tensor`. The value takes` tensor_x [index] `as an example, and the assignment takes` tensor_x [index] = u` as an example for detailed description. Among them, tensor_x is a `Tensor`, which is sliced; index means the index, u means the assigned value, which can be` scalar` or `Tensor (size = 1)`. The index types are as follows:

- Slice index: index is `slice`
  - Value: `tensor_x[start: stop: step]`, where Slice (start: stop: step) has the same syntax as Python, and will not be repeated here.
  - Assignment: `tensor_x[start: stop: step] = u`.

- Ellipsis index: index is `ellipsis`
  - Value: `tensor_x [...]`.
  - Assignment: `tensor_x [...] = u`.

- Boolean constant index: index is `True`, index is `False` is not supported temporarily.
  - Value: `tensor_x[True]`.
  - Assignment: Not supported yet.
  
- Tensor index: index is `Tensor`
   - Value: `tensor_x [index]`, `index` must be `Tensor` of data type `int32` or `int64`,
     the element value range is `[0, tensor_x.shape[0])`.
   - Assignment: `tensor_x [index] = U`.
      - `tensor_x` data type must be one of the following: `float16`, `float32`, `int8`, `uint8`.
      - `index` must be `Tensor` of data type `int32`, the element value range is `[0, tensor_x.shape [0])`.
      - `U` can be `Number`, `Tensor`, `Tuple` only containing `Number`, `Tuple` only containing `Tensor`.
        - Single `Number` or every `Number` in  `Tuple` must be the same type as `tensor_x`, ie
          When the data type of `tensor_x` is `uint8` or `int8`, the `Number` type should be `int`;
          When the data type of `tensor_x` is `float16` or `float32`, the `Number` type should be `float`.
        - Single `Tensor` or every `Tensor in Tuple` must be consistent with the data type of `tensor_x`,
          when single `Tensor`, the `shape` should be equal to or broadcast as `index.shape + tensor_x.shape [1:]`.
        - `Tuple` containing `Number` must meet requirement:
          `len (Tuple) = (index.shape + tensor_x.shape [1:]) [-1]`.
        - `Tuple` containing `Tensor` must meet requirements:
          the `shape` of each `Tensor` should be the same,
          `(len (Tuple),) + Tensor.shape` should be equal to or broadcast as `index.shape + tensor_x.shape [1:]`.

- None constant index: index is `None`
  - Value: `tensor_x[None]`, results are consistent with numpy.
  - Assignment: Not supported yet.

- tuple index: index is `tuple`
  - The tuple element is a slice:
    - Value: for example `tensor_x[::,: 4, 3: 0: -1]`.
    - Assignment: for example `tensor_x[::,: 4, 3: 0: -1] = u`.
  - The tuple element is Number:
    - Value: for example `tensor_x[2,1]`.
    - Assignment: for example `tensor_x[1,4] = u`.
  - The tuple element is a mixture of slice and ellipsis:
    - Value: for example `tensor_x[..., ::, 1:]`.
    - Assignment: for example `tensor_x[..., ::, 1:] = u`.
  - Not supported in other situations

In addition, tuple also supports slice value operation, `tuple_x [start: stop: step]`, which has the same effect as Python, and will not be repeated here.

### Unsupported Syntax

Currently, the following syntax is not supported in network constructors: 
 `raise`, `yield`, `async for`, `with`, `async with`, `assert`, `import`, and `await`.

## Network Definition Constraints

### Instance Types on the Entire Network
* Common Python function with the [@ms_function](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.html#mindspore.ms_function) decorator.
* Cell subclass inherited from [nn.Cell](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell).

### Network Input Type
* The training data input parameters of the entire network must be of the Tensor type.
* The generated ANF diagram cannot contain the following constant nodes: string constants, constants with nested tuples, and constants with nested lists.

### Network Graph Optimization
 During graph optimization at the ME frontend, the dataclass, dictionary, list, and key-value pair types are converted to tuple types, and the corresponding operations are converted to tuple operations.

### Network Construction Components

| Category                              | Content
| :-----------                          |:--------
| `Cell` instance                       |[mindspore/nn/*](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.nn.html), and custom [Cell](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell).
| Member function of a `Cell` instance  | Member functions of other classes in the construct function of Cell can be called.
| Function                              | Custom Python functions and system functions listed in the preceding content.
| Dataclass instance                    | Class decorated with @dataclass.
| Primitive operator                    |[mindspore/ops/operations/*](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.ops.operations.html).
| Composite operator                    |[mindspore/ops/composite/*](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.ops.composite.html).
| Operator generated by constexpr       |Uses the value generated by [@constexpr](https://www.mindspore.cn/api/en/r0.6/api/python/mindspore/mindspore.ops.html#mindspore.ops.constexpr) to calculate operators.


### Other Constraints
Input parameters of the construct function on the entire network and parameters of functions modified by the ms_function decorator are generalized during the graph compilation. Therefore, they cannot be transferred to operators as constant input. Therefore, in graph mode, the parameter passed to the entry network can only be Tensor. As shown in the following example:
* The following is an example of incorrect input:
    ```python
    class ExpandDimsTest(Cell):
        def __init__(self):
            super(ExpandDimsTest, self).__init__()
            self.expandDims = P.ExpandDims()

        def construct(self, input_x, input_axis):
            return self.expandDims(input_x, input_axis)
    expand_dim = ExpandDimsTest()
    input_x = Tensor(np.random.randn(2,2,2,2).astype(np.float32))
    expand_dim(input_x, 0)
    ```
    In the example, ExpandDimsTest is a single-operator network with two inputs: input_x and input_axis. The second input of the ExpandDims operator must be a constant. This is because input_axis is required when the output dimension of the ExpandDims operator is deduced during graph compilation. As the network parameter input, the value of input_axis is generalized into a variable and cannot be determined. As a result, the output dimension of the operator cannot be deduced, causing the graph compilation failure. Therefore, the input required by deduction in the graph compilation phase must be a constant. In APIs, the "constant input is needed" is marked for parameters that require constant input of these operators.

* Directly enter the needed value or a member variable in a class for the constant input of the operator in the construct function. The following is an example of correct input:
    ```python
    class ExpandDimsTest(Cell):
        def __init__(self, axis):
            super(ExpandDimsTest, self).__init__()
            self.expandDims = P.ExpandDims()
            self.axis = axis

        def construct(self, input_x):
            return self.expandDims(input_x, self.axis)
    axis = 0
    expand_dim = ExpandDimsTest(axis)
    input_x = Tensor(np.random.randn(2,2,2,2).astype(np.float32))
    expand_dim(input_x)
    ```
