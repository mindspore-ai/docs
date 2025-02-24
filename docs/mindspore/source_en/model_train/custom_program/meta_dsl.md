# Custom operator combination

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/custom_program/meta_dsl.md)

## Overview

For operators implemented by a combination of multiple small operators, the traditional development process requires defining primitive registration, type derivation, automatic differentiation, and small operator combination logic for different hardware platforms to implement the operator. In order to reduce operator development costs and improve development efficiency, MindSpore provides the Meta DSL programming paradigm.

Meta DSL programming paradigm is an efficient programming paradigm designed for operator development. It is applied to static graph mode. It provides an efficient and concise operator development method through C++ composition. After realizing the primitive registration of the operator, the operator combination logic of the operator can be developed through the Meta DSL programming paradigm, and then the development of the operator is completed. There is no need to define type derivation, automatic differential, or the different operator combination logic for different platforms.

This guide is designed to help developers fully understand the functional interfaces and reference examples of the Meta DSL programming paradigm.

## Functional interfaces

### REGISTER_FUNCTION_OP

Function: Register an operator, indicating that the operator is implemented by combining small operators.

Parameter:

1. Operator name: The name of the operator.
2. Verification function of operator parameters: optional parameter, `nullptr` is allowed to be passed in. Used to verify whether the operator's inputs are legal, such as their type and shape.
3. Customized Backward: optional parameter, used to define the name of the operator's backward. Custom backward is also implemented through the Meta DSL programming paradigm.

Example:

```cpp
REGISTER_FUNCTION_OP(CustomOp)
REGISTER_FUNCTION_OP(CustomOp, OpCheckFunction)
REGISTER_FUNCTION_OP(CustomOp, OpCheckFunction, CustomOpGrad)
```

### BeginFunction

Function: Start operator construction.

Parameter: Operator name, inputs.

Example: `BeginFunction(CustomOp, input0, input1, input2, ...)`

### EndFunction

Function: End operator construction.

Parameter: Operator name.

Example: `EndFunction(CustomOp)`

### Prim

Function: Primitive.

Parameter: Operator name.

Example: `auto op = Prim(Add);`

### Value

Function: Constant value.

Parameter: int, float, bool, char*, ValuePtr and other input types.

Example: `Value(0)`, `Value(1.0)`, `Value(true)`, `Value("valid")`, `Value<int32_t>(100)`

### Call

Function: Logic of input and output of single operator.

Parameter: The number of inputs can be 1 or more.

Example: `Call(Prim(Add), x, y)`

### Return

Function: Return the output result.

Parameter: The output of Parameter.

Example: `Return(out)`

### If

Function: Control flow `if-else` expression.

Parameter:

1. Condition: The condition of if.
2. True branch: The lambda function needs to be passed in.
3. False branch: The lambda function needs to be passed in.
4. Inputs: The inputs are enclosed by parentheses (), which contains all the external variables needed by the true branch and false branch.

Example:

```cpp
auto condition = Call(Prim(Equal), x, Value(2));
auto true_branch = [&]() { Return(x); };
auto false_branch = [&]() { Return(Value(y)); };
auto out = If(condition, true_branch, false_branch, (x, y))
```

The corresponding Python code is:

```python
if x == 2:
  return x
return y
```

### For

Function: For loop in control flow. Please refer to `mindspore.ops.ForiLoop` for more details.

Parameter:

1. lower: The start index of loop.
2. upper: The end index of loop.
3. loop_func: The loop function, takes two arguments.
4. init_val:The init value. Supports Tensor, number, str, bool, list, tuple, dict.

Example:

```cpp
auto cumsum = [&](const NodePtr &index, const NodePtr &res) { Return(Call(Prim(Add), index, res)); };
auto out = For(lower, upper, cumsum, x);
```

The corresponding Python code is:

```python
def cumsum(index, res):
  return index + res

for i in range(lower, upper):
  x = cumsum(i, x)
return x
```

### While

Function: While loop in control flow. Please refer to `mindspore.ops.WhileLoop` for more details.

Parameter:

1. cond_func: The condition function, take one argument.
2. loop_func: The loop function, take one argument and return value has the same type with input argument.
3. init_val: The initial value. Supports Tensor, number, str, bool, list, tuple, dict.

Example:

```cpp
auto cond_func = [&](const NodePtr &x) { Return(Less(x, Value(100))); };
auto loop_func = [&](const NodePtr &x) { Return(Call(Prim(Add), x, Value(1))); };
auto out = While(cond_func, loop_func, x);
```

The corresponding Python code is:

```python
while x < 100:
  x = x + 1
return x
```

### Scan

Function: Scan a function over an array while the processing of the current element depends on the execution result of the previous element. Please refer to  `mindspore.ops.Scan` for more details.

Parameter:

1. loop_func: The loop function, takes two arguments.
2. init: An initial loop carry value. Supports Tensor, number, str, bool, list, tuple, dict.
3. xs: The value over which to scan. Supports list, tuple, None
4. length: Optional. The length of the array xs. The default value is ``None``.

Example:

```cpp
auto cumsum = [&](const NodePtr &input, const NodePtr &elem) {
  auto out = Call(Prim(Add), input, elem);
  Return(Tuple(out, out));
};
auto [res, ys] = Scan(cumsum, init, xs);
```

The corresponding Python code is:

```python
def cumsum(input, elem):
  out = input + elem
  return out, out

res = init
ys = []
for elem in xs:
  res, y = cumsum(res, elem)
  ys.append(y)
return res, ys
```

### `Tuple`

Function: Create a tuple.

Parameter: 0 or more elements.

Example: `Tuple(x, y, z)`

### `List`

Function: Create a list.

Parameter: 0 or more elements.

Example: `List(x, y, z)`

### `Raise`

Function: Used to throw exceptions.

Parameter: Exception type and exception message.

Example: `Raise("ValueError", "Not supported")`

### `IsInstance`

Function: Determine the input type.

Parameter: input and type fro comparison. The latter is the `TypeId` enumeration type, such as `TypeId::kObjectTypeTensorType`、`TypeId::kNumberTypeInt`、`TypeId::kNumberTypeFloat`、`TypeId::kNumberTypeBool`.

Example: `IsInstance(x, TypeId::kObjectTypeTensorType)`, `IsInstance(x, {TypeId::kNumberTypeInt, TypeId::kNumberTypeFloat})`

### `GetItem`

Function: Get value according to index, corresponding to `x[y]`.

Example: `GetItem(x, y)`

### `SetItem`

Function: Assign value according to index, corresponding to `x[y] = z`.

Example: `SetItem(x, y, z)`

### `Equal`

Function: Logical judgment operation, equal to.

Example: `Equal(x, y)`

### `NotEqual`

Function: Logical judgment operation, not equal.

Example: `NotEqual(x, y)`

### `Greater`

Function: Logical judgment operation, greater than.

Example: `Greater(x, y)`

### `Less`

Function: Logical judgment operation, less than.

Example: `Less(x, y)`

### `GreaterEqual`

Function: Logical judgment operation, greater than or equal to.

Example: `GreaterEqual(x, y)`

### `LessEqual`

Function: Logical judgment operation, less than or equal to.

Example: `LessEqual(x, y)`

### `IsNone`

Function: Determine whether the input is None.

Example: `IsNone(x)`

### `IsNotNone`

Function: Determine whether the input is not None.

Example: `IsNotNone(x)`

### `And`

Function: logic AND.

Example: `And(x, y)`

### `Or`

Function: logic OR.

Example: `Or(x, y)`

### `Not`

Function: logic NOT.

Example: `Not(x)`

## Reference examples

### Simple scenario

Assume that operator `CustomOp` has three inputs `x`, `y`, `z`, which is composed of operator `A` and operator `B`.

Register the operator through `REGISTER_FUNCTION_OP(CustomOp)`, and then define the operator combination logic of operator `CustomOp`. The code example is as follows:

```cpp
BeginFunction(CustomOp, x, y, z) {
  auto value = Call(Prim(A), x, y);
  auto out = Call(Prim(B), value, z);
  Return(out);
}
EndFunction(CustomOp)
```

### Control flow if-else scenario

Assume that operator `CustomOp` has two inputs `x` and `y`, which is composed of operator `A` and operator `B`.

Register the operator through `REGISTER_FUNCTION_OP(CustomOp)`, and then define the operator combination logic of operator `CustomOp`. The code example is as follows:

```cpp
BeginFunction(CustomOp, x, y) {
  auto true_branch = [&]() { Return(Call(Prim(A), x)); }
  auto false_branch = [&]() { Return(Call(Prim(B), y)); }
  auto condition = Less(x, Value(2));
  Return(If(condition, true_branch, false_branch, (x, y)));
}
EndFunction(CustomOp)
```

### Complex scenario

Take the operator `Dense` as an example. It is composed of multiple small operators.

Register the operator through `REGISTER_FUNCTION_OP(Dense, CheckFunc, DenseGrad)`. The code example is as follows:

```cpp
BeginFunction(Dense, input, weight, bias) {
  auto get_transpose_perm = [&](const NodePtr &weight) {
    auto size = Call(Prim(Rank), weight);
    auto true_branch = [&]() {
      auto perm = Call(Prim(MakeRange), size);
      perm = Call(Prim(TupleSetItem), perm, Value(0), Value(0));
      Return(perm);
    };
    auto false_branch = [&]() {
      auto perm = Call(Prim(MakeRange), size);
      auto minus_one = Call(Prim(ScalarSub), size, Value(1));
      auto minus_two = Call(Prim(ScalarSub), size, Value(2));
      perm = Call(Prim(TupleSetItem), perm, minus_one, minus_two);
      perm = Call(Prim(TupleSetItem), perm, minus_two, minus_one);
      Return(perm);
    };
    return If(Less(size, Value(2)), true_branch, false_branch, (size));
  };

  auto perm = get_transpose_perm(weight);
  auto weight_transposed = Call(Prim(Transpose), weight, perm);
  auto contiguous_out = Call(Prim(Contiguous), weight_transposed);
  auto output = Call(Prim(MatMulExt), input, contiguous_out);
  auto true_branch = [&]() { Return(Call(Prim(Add), output, bias)); };
  auto false_branch = [&]() { Return(output); };
  Return(If(IsNotNone(bias), true_branch, false_branch, (output, bias)));
}
EndFunction(Dense)

void CheckFunc(const PrimitivePtr &primitive, const AbstractBasePtrList &inputs) {
  // The implementation code for the function that validates the type and shape of Dense's inputs is omitted here.
}

BeginFunction(DenseGrad, input, weight, bias, out, dout) {
  // The implementation code for DenseGrad is omitted here.
}
EndFunction(DenseGrad)
```
