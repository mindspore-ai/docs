# Constraints on Network Construction Using Python

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

| Operation          | Description
| :-----------    |:--------
| Unary operator      |`+`,`-`, and`not`. The operator `+` supports only scalars.
| Binary operator      |`+`, `-`, `*`, `/`, and `%`.
| `if` expression      | For example, `a = x if x < y else y`.
| Comparison expression      | `>`, `>=`, `<`, `<=`, `==`, and `! =`.
| Logical expression      | `and` and `or`.
| `lambda` expression  | For example, `lambda x, y: x + y`.
| Reserved keyword type   | `True`, `False`, and `None`.

### Statement Types

| Statement         | Compared with Python
| :----------- |:--------
| `for`        | Nested for loops are partially supported. Iteration sequences must be tuples or lists.
| `while`      | Nested while loops are partially supported.
| `if`         | Same as that in Python. The input of the `if` condition must be a constant.
| `def`        | Same as that in Python.
| Assignment statement     | Accessed multiple subscripts of lists and dictionaries cannot be used as l-value.

### System Functions

* len
* partial
* map
* zip
* range

### Function Parameters

* Default parameter value: The data types `int`, `float`, `bool`, `None`, `str`, `tuple`, `list`, and `dict` are supported, whereas `Tensor` is not supported.
* Variable parameter: Functions with variable parameters cannot be used for backward propagation on computational graphs.
* Key-value pair parameter: Functions with key-value pair parameters cannot be used for backward propagation on computational graphs.
* Variable key-value pair parameter: Functions with variable key-value pairs cannot be used for backward propagation on computational graphs.

### Operators

| Operator         | Supported Type
| :----------- |:--------
| `+`          |Scalar, `Tensor`, and `tuple`
| `-`          |Scalar and `Tensor`
| `*`          |Scalar and `Tensor`
| `/`          |Scalar and `Tensor`
| `[]`         |The operation object type can be `list`, `tuple`, or `Tensor`. Accessed multiple subscripts of lists and dictionaries can be used as r-values instead of l-values. The index type cannot be Tensor. For details about access constraints for the tuple and Tensor types, see the description of slicing operations.

### Slicing Operations

* `tuple` slicing operation: `tuple_x[start:stop:step]`
    * `tuple_x` indicates a tuple on which the slicing operation is performed.
    * `start`: index where the slice starts. The value is of the `int` type, and the value range is `[-length(tuple_x), length(tuple_x) - 1]`. Default values can be used. The default settings are as follows:
        * When `step > 0`, the default value is `0`.
        * When `step < 0`, the default value is `length(tuple_x) - 1`.
    * `end`: index where the slice ends. The value is of the `int` type, and the value range is `[-length(tuple_x) - 1, length(tuple_x)]`. Default values can be used. The default settings are as follows:
        * When `step > 0`, the default value is `length(tuple_x)`.
        * When `step < 0`, the default value is `-1`.
    * `step`: slicing step. The value is of the `int` type, and its range is `step! = 0`. The default value `1` can be used.

* `Tensor` slicing operation: `tensor_x[start0:stop0:step0, start1:stop1:step1, start2:stop2:step2]`
    * `tensor_x` indicates a `Tensor` with at least three dimensions. The slicing operation is performed on it.
    * `start0`: index where the slice starts in dimension 0. The value is of the `int` type. Default values can be used. The default settings are as follows:
        * When `step > 0`, the default value is `0`.
        * When `step < 0`, the default value is `-1`.
    * `end0`: index where the slice ends in dimension 0. The value is of the `int` type. Default values can be used. The default settings are as follows:
        * When `step > 0`, the default value is `length(tuple_x)`.
        * When `step < 0`, the default value is `-(1 + length(tuple_x))`.
    * `step0`: slicing step in dimension 0. The value is of the `int` type, and its range is `step! = 0`. The default value `1` can be used.
    * If the number of dimensions for slicing is less than that for `Tensor`, all elements are used by default if no slice dimension is specified.
    * Slice dimension reduction operation: If an integer index is transferred to a dimension, the elements of the corresponding index in the dimension is obtained and the dimension is eliminated. For example, after `tensor_x[2:4:1, 1, 0:5:2]` with shape (4, 3, 6) is sliced, a `Tensor` with shape (2, 3) is generated. The first dimension of the original `Tensor` is eliminated.

### Unsupported Syntax

Currently, the following syntax is not supported in network constructors:
 `break`, `continue`, `pass`, `raise`, `yield`, `async for`, `with`, `async with`, `assert`, `import`, and `await`.

## Network Definition Constraints

### Instance Types on the Entire Network

* Common Python function with the [@ms_function](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.html#mindspore.ms_function) decorator.
* Cell subclass inherited from [nn.Cell](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell).

### Network Input Type

* The training data input parameters of the entire network must be of the Tensor type.
* The generated ANF diagram cannot contain the following constant nodes: string constants, constants with nested tuples, and constants with nested lists.

### Network Graph Optimization

 During graph optimization at the ME frontend, the dataclass, dictionary, list, and key-value pair types are converted to tuple types, and the corresponding operations are converted to tuple operations.

### Network Construction Components

| Category                 | Content
| :-----------         |:--------
| `Cell` instance           |[mindspore/nn/*](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.nn.html), and custom [Cell](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell).
| Member function of a `Cell` instance | Member functions of other classes in the construct function of Cell can be called.
| Function                 | Custom Python functions and system functions listed in the preceding content.
| Dataclass instance        | Class decorated with @dataclass.
| Primitive operator        |[mindspore/ops/operations/*](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.ops.operations.html).
| Composite operator        |[mindspore/ops/composite/*](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.ops.composite.html).
| Operator generated by constexpr     |Uses the value generated by [@constexpr](https://www.mindspore.cn/api/en/0.1.0-alpha/api/python/mindspore/mindspore.ops.html#mindspore.ops.constexpr) to calculate operators.

### Other Constraints

Input parameters of the construct function on the entire network and parameters of functions modified by the ms_function decorator are generalized during the graph compilation. Therefore, they cannot be transferred to operators as constant input, as shown in the following example:

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
