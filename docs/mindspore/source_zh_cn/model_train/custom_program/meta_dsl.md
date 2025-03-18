# 自定义算子拼接

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/custom_program/meta_dsl.md)

## 概述

对于多个小算子拼接组合实现的算子，传统的算子开发流程下需要定义实现该算子的原语注册、类型推导逻辑、自动微分反向逻辑、不同硬件平台的小算子拼接逻辑。为了降低算子开发成本并提高开发效率，MindSpore框架提供了Meta DSL编程范式。

Meta DSL编程范式是一款为算子开发而设计的高效编程范式，应用于静态图模式，通过C++构图提供了高效简洁的算子开发方式。在实现算子的原语注册后，通过Meta DSL编程范式开发该算子的小算子拼接逻辑，就能完成该算子的整体功能开发。不需要定义算子的类型推导逻辑、自动微分反向逻辑，也不需要针对不同平台编写不同的小算子拼接逻辑。

本指南旨在帮助开发者全面了解Meta DSL编程范式的功能接口和使用示例。

## 功能接口

### REGISTER_FUNCTION_OP

功能：注册算子，表示该算子是由小算子拼接组合实现。

参数：

1. 算子名：算子的名字。
2. 算子参数的校验函数：可选参数，允许传入 `nullptr`。用于校验算子输入是否合法，比如输入的类型和shape。
3. 自定义反向：可选参数，用于定义算子反向的名字。自定义反向也是通过 Meta DSL 编程范式实现的。

示例：

```cpp
REGISTER_FUNCTION_OP(CustomOp)
REGISTER_FUNCTION_OP(CustomOp, OpCheckFunction)
REGISTER_FUNCTION_OP(CustomOp, OpCheckFunction, CustomOpGrad)
```

### BeginFunction

功能：开始算子拼接。

参数：算子名、算子输入。

示例：`BeginFunction(CustomOp, input0, input1, input2, ...)`

### EndFunction

功能：结束算子拼接。

参数：算子名。

示例：`EndFunction(CustomOp)`

### Prim

功能：表示算子原语。

参数：算子名。

示例：`auto op = Prim(Add);`

### Value

功能：表示常量值。

参数：int、float、bool、char*、ValuePtr等输入类型。

示例：`Value(0)`, `Value(1.0)`, `Value(true)`, `Value("valid")`, `Value<int32_t>(100)`

### Call

功能：表示单个算子的输入输出逻辑。

参数：输入数量可以是1个或者多个。

示例：`Call(Prim(Add), x, y)`

### Return

功能：表示输出。

参数：算子的输出。

示例：`Return(out)`

### If

功能：控制流 `if-else` 表达式。

参数：

1. 条件：if的条件。
2. true分支：需要传入lambda函数。
3. false分支：需要传入lambda函数。
4. 输入：使用圆括号()包起来，里面放的是true分支和false分支需要用到的所有外部变量。

示例：

```cpp
auto condition = Call(Prim(Equal), x, Value(2));
auto true_branch = [&]() { Return(x); };
auto false_branch = [&]() { Return(Value(y)); };
auto out = If(condition, true_branch, false_branch, (x, y))
```

对应的Python源码是：

```python
if x == 2:
  return x
return y
```

### For

功能：控制流for循环。详情请参考 `mindspore.ops.ForiLoop`。

参数：

1. lower：循环的起始索引值。
2. upper：循环的结束索引值。
3. loop_func：循环体函数，接受两个参数。
4. init_val：循环的初始值。支持 Tensor、number、str、bool、list、tuple、dict。

示例：

```cpp
auto cumsum = [&](const NodePtr &index, const NodePtr &res) { Return(Call(Prim(Add), index, res)); };
auto out = For(lower, upper, cumsum, x);
```

对应的Python源码是：

```python
def cumsum(index, res):
  return index + res

for i in range(lower, upper):
  x = cumsum(i, x)
return x
```

### While

功能：控制流while循环。详情请参考 `mindspore.ops.WhileLoop`。

参数：

1. cond_func：循环的条件函数，接受一个参数。
2. loop_func：循环体函数，接受一个参数，并且返回值与输入参数的类型相同。
3. init_val：循环的初始值。支持Tensor、number、str、bool、list、tuple、dict。

示例：

```cpp
auto cond_func = [&](const NodePtr &x) { Return(Less(x, Value(100))); };
auto loop_func = [&](const NodePtr &x) { Return(Call(Prim(Add), x, Value(1))); };
auto out = While(cond_func, loop_func, x);
```

对应的Python源码是:

```python
while x < 100:
  x = x + 1
return x
```

### Scan

功能：将一个函数循环作用于一个数组，且对当前元素的处理依赖上一个元素的执行结果。详情请参考 `mindspore.ops.Scan`。

参数：

1. loop_func：循环体函数，接受两个参数。
2. init：循环的初始值。支持Tensor、number、str、bool、list、tuple、dict。
3. xs：用于执行循环扫描的数组。支持list、tuple、None。
4. length：可选。数组xs的长度，默认值为 ``None``。

示例：

```cpp
auto cumsum = [&](const NodePtr &input, const NodePtr &elem) {
  auto out = Call(Prim(Add), input, elem);
  Return(Tuple(out, out));
};
auto output = Scan(cumsum, init, xs);
auto res = GetItem(output, Value(0));
auto ys = GetItem(output, Value(1));
```

对应的Python源码是：

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

功能：创建Tuple元组。

参数：0个或多个元素。

示例：`Tuple(x, y, z)`

### `List`

功能：创建List列表。

参数：0个或多个元素。

示例：`List(x, y, z)`

### `Raise`

功能：用于抛出异常。

参数：报错类型、报错信息。

示例：`Raise("ValueError", "Not supported")`

### `IsInstance`

功能：判断输入的类型。

参数：输入、目标类型。后者是 `TypeId` 枚举类型，例如 `TypeId::kObjectTypeTensorType`、`TypeId::kNumberTypeInt`、`TypeId::kNumberTypeFloat`、`TypeId::kNumberTypeBool`等。

示例：`IsInstance(x, TypeId::kObjectTypeTensorType)`, `IsInstance(x, {TypeId::kNumberTypeInt, TypeId::kNumberTypeFloat})`

### `GetItem`

功能：索引取值，对应 `x[y]`。

示例：`GetItem(x, y)`

### `SetItem`

功能：索引赋值，对应 `x[y] = z`。

示例：`SetItem(x, y, z)`

### `Equal`

功能：逻辑判断，等于。

示例：`Equal(x, y)`

### `NotEqual`

功能：逻辑判断，不等于。

示例：`NotEqual(x, y)`

### `Greater`

功能：逻辑判断，大于。

示例：`Greater(x, y)`

### `Less`

功能：逻辑判断，小于。

示例：`Less(x, y)`

### `GreaterEqual`

功能：逻辑判断，大于等于。

示例：`GreaterEqual(x, y)`

### `LessEqual`

功能：逻辑判断，小于等于。

示例：`LessEqual(x, y)`

### `IsNone`

功能：判断输入是None。

示例：`IsNone(x)`

### `IsNotNone`

功能：判断输入不是None。

示例：`IsNotNone(x)`

### `And`

功能：逻辑与。

示例：`And(x, y)`

### `Or`

功能：逻辑或。

示例：`Or(x, y)`

### `Not`

功能：逻辑非。

示例：`Not(x)`

## 使用示例

### 简单拼接场景

假定算子 `CustomOp` 有3个输入 `x`、`y`、`z`，它由小算子`A`和小算子`B`拼接组合而成。

通过 `REGISTER_FUNCTION_OP(CustomOp)` 注册算子，然后定义算子 `CustomOp` 的算子拼接逻辑，代码示例如下：

```cpp
BeginFunction(CustomOp, x, y, z) {
  auto value = Call(Prim(A), x, y);
  auto out = Call(Prim(B), value, z);
  Return(out);
}
EndFunction(CustomOp)
```

### 控制流 if-else 场景

假定算子 `CustomOp` 有2个输入 `x`、`y`，它由小算子`A`和小算子`B`拼接组合而成。

通过 `REGISTER_FUNCTION_OP(CustomOp)` 注册算子，然后定义算子 `CustomOp` 的算子拼接逻辑，代码示例如下：

```cpp
BeginFunction(CustomOp, x, y) {
  auto true_branch = [&]() { Return(Call(Prim(A), x)); }
  auto false_branch = [&]() { Return(Call(Prim(B), y)); }
  auto condition = Less(x, Value(2));
  Return(If(condition, true_branch, false_branch, (x, y)));
}
EndFunction(CustomOp)
```

### 复杂场景

以 `Dense` 算子的拼接逻辑为例，它由多个小算子拼接组合而成。

通过 `REGISTER_FUNCTION_OP(Dense, CheckFunc, DenseGrad)` 注册算子，其实现逻辑的代码示例如下：

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
