# Python源码构造网络约束

## 概述

  MindSpore完成从用户源码到计算图的编译，用户源码基于Python语法编写，当前MindSpore支持将普通函数或者继承自nn.Cell的实例转换生成计算图，暂不支持将任意Python源码转换成计算图，所以对于用户源码支持的写法有所限制，主要包括语法约束和网络定义约束两方面。随着MindSpore的演进，这些约束可能会发生变化。

## 语法约束

### 支持的Python数据类型

* Number：包括`int`、`float`、`bool`，不支持复数类型。
* String
* List：当前只支持append方法；List的更新会拷贝生成新的List。
* Tuple
* Dictionary：当前`key`只支持String类型

### MindSpore扩展数据类型

* Tensor：Tensor变量必须是已定义实例。

### 表达式类型

| 操作名          | 具体操作
| :-----------    |:--------
| 一元操作符      |`+`、`-`、`not`，其中`+`操作符只支持标量。
| 二元操作符      |`+`、`-`、`*`、`/`、`%`
| `if`表达式      |例如`a = x if x < y else y`。
| 比较表达式      | `>`、`>=`、`<`、`<=`、`==`、`!=`
| 逻辑表达式      | `and`、 `or`
| `lambda`表达式  | 例如`lambda x, y: x + y`。
| 保留关键字类型   | `True`、`False`、`None`

### 语句类型

| 语句         | 与Python对比
| :----------- |:--------
| `for`        | 迭代序列必须是Tuple/List，部分嵌套场景支持。
| `while`      | 部分嵌套场景支持。
| `if`         | 与Python使用原则一致，但if条件的输入只支持常量。
| `def`        | 相同。
| 赋值语句      | List和Dictionary的多重下标访问不支持作为左值。

### 系统函数

* len
* partial
* map
* zip
* range

### 函数参数

* 参数默认值：目前不支持默认值设为`Tensor`类型数据，支持`int`、`float`、`bool`、`None`、`str`、`tuple`、`list`、`dict`类型数据。
* 可变参数：目前不支持带可变参数的函数求反向。
* 键值对参数：目前不支持带键值对参数的函数求反向。
* 可变键值对参数：目前不支持带可变键值对的函数求反向。

### 操作符

| 运算符         | 支持类型
| :----------- |:--------
| `+`          |标量、`Tensor`、`tuple`
| `-`          |标量、`Tensor`
| `*`          |标量、`Tensor`
| `/`          |标量、`Tensor`
| `[]`         |操作对象类型支持`list`、`tuple`、`Tensor`，支持多重下标访问作为右值，但不支持多重下标访问作为左值，且索引类型不支持Tensor；Tuple、Tensor类型访问限制见切片操作中的说明。

### 切片操作

* `tuple`切片操作：`tuple_x[start:stop:step]`
    * `tuple_x`为一个元组，是被执行切片操作的目标。
    * `start`：切片的起始位置索引，类型为`int`，取值范围为`[-length(tuple_x), length(tuple_x) - 1]`。可缺省，缺省配置如下：
        * 当`step > 0`时，缺省值为`0`。
        * 当`step < 0`时，缺省值为`length(tuple_x) - 1`。
    * `end`：切片的结束位置索引，类型为`int`，取值范围为`[-length(tuple_x) - 1, length(tuple_x)]`。可缺省，缺省配置如下：
        * 当`step > 0`时，缺省值为`length(tuple_x)`。
        * 当`step < 0`是，缺省值为`-1`。
    * `step`：切片的步长，类型为`int`，取值范围为`step != 0`。可缺省，缺省值为`1`。

* `Tensor`切片操作：`tensor_x[start0:stop0:step0, start1:stop1:step1, start2:stop2:step2]`
    * `tensor_x`是一个维度不低于3维的`Tensor`，对其进行切片操作。
    * `start0`：在第0维上进行切片的起始位置索引，类型为`int`，可缺省，缺省配置如下：
        * 当`step > 0`时，缺省值为`0`。
        * 当`step < 0`时，缺省值为`-1`。
    * `end0`：在第0维上进行切片的结束位置索引，类型为`int`，可缺省，缺省配置如下：
        * 当`step > 0`时，缺省值为`length(tuple_x)`。
        * 当`step < 0`是，缺省值为`-(1 + length(tuple_x))`。
    * `step0`：在第0维上进行切片的步长，类型为`int`，取值范围为`step != 0`。可缺省，缺省值为`1`。
    * 如果进行切片的维数少于`Tensor`的维数，则未指定切片的维度默认取全部元素。
    * 切片降维操作：在某维度上传入整数索引，则取出该维度上对应索引的元素，且消除该维度，如shape为(4, 3, 6)的`tensor_x[2:4:1, 1, 0:5:2]`切片之后，生成一个shape为(2, 3)的`Tensor`，原`Tensor`的第1维被消除。

### 不支持的语法

目前在网络构造函数里面暂不支持以下语法：
 `break`、 `continue`、 `pass`、 `raise`、 `yield`、 `async for`、 `with`、 `async with`、 `assert`、 `import`、 `await`。

## 网络定义约束

### 整网实例类型

* 带[@ms_function](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.html#mindspore.ms_function)装饰器的普通Python函数。
* 继承自[nn.Cell](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell)的Cell子类。

### 网络输入类型

* 整网的训练数据输入参数只能是Tensor类型。
* 生成的ANF图里面不能包含这几种常量节点：字符串类型常量、带有Tuple嵌套的常量、带有List嵌套的常量。

### 网络图优化

 在ME前端图优化过程中，会将DataClass类型、Dictionary、List、键值对操作转换为Tuple相关操作。

### 网络构造组件

| 类别                 | 内容
| :-----------         |:--------
| `Cell`实例           |[mindspore/nn/*](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.nn.html)、自定义[Cell](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell)。
| `Cell`实例的成员函数 | Cell的construct中可以调用其他类成员函数。
| 函数                 | 自定义Python函数、前文中列举的系统函数。
| dataclass实例        | 使用@dataclass装饰的类。
| Primitive算子        |[mindspore/ops/operations/*](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.ops.operations.html)
| Composite算子        |[mindspore/ops/composite/*](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.ops.composite.html)
| constexpr生成算子    |使用[@constexpr](https://www.mindspore.cn/api/zh-CN/0.1.0-alpha/api/python/mindspore/mindspore.ops.html#mindspore.ops.constexpr)生成的值计算算子。

### 其他约束

整网construct函数输入的参数以及使用ms_function装饰器修饰的函数的参数在图编译过程中会进行泛化，不能作为常量输入传给算子使用，如下例所示：

* 错误的写法如下：

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

    在示例中，ExpandDimsTest是一个只有单算子的网络，网络的输入有input_x和input_axis两个。因为ExpandDims算子的第二个输入需要是常量，这是因为在图编译过程中推导ExpandDims算子输出维度的时候需要用到，而input_axis作为网络参数输入会泛化成变量，无法确定其值，从而无法推导算子的输出维度导致图编译失败。所以在图编译阶段需要值推导的输入都应该是常量输入。在API中，这类算子需要常量输入的参数会进行说明，标注"constant input is needed"。

* 正确的写法是在construct函数里面对算子的常量输入直接填入需要的值或者是一个类的成员变量，如下：

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
