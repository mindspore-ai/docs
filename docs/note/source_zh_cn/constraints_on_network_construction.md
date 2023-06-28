# Python源码构造网络约束

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/constraints_on_network_construction.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

  MindSpore完成从用户源码到计算图的编译，用户源码基于Python语法编写，当前MindSpore支持将普通函数或者继承自nn.Cell的实例转换生成计算图，暂不支持将任意Python源码转换成计算图，所以对于用户源码支持的写法有所限制，主要包括语法约束和网络定义约束两方面。随着MindSpore的演进，这些约束可能会发生变化。

## 语法约束

### 支持的Python数据类型

- Number：包括`int`、`float`、`bool`，不支持复数类型。
- String
- List：当前只支持append方法；List的更新会拷贝生成新的List。
- Tuple
- Dictionary：当前`key`只支持String类型

### MindSpore扩展数据类型

- Tensor：Tensor变量必须是已定义实例。

### 表达式类型

| 操作名          | 具体操作
| :-----------   |:--------
| 一元操作符       |`+`、`-`、`not`，其中`+`操作符只支持标量。
| 数学表达式       |`+`、`-`、`*`、`/`、`%`、`**`、`//`
| `if`表达式      |例如`a = x if x < y else y`。
| 比较表达式       | `>`、`>=`、`<`、`<=`、`==`、`!=`
| 逻辑表达式       | `and`、 `or`
| `lambda`表达式  | 例如`lambda x, y: x + y`。
| 保留关键字类型    | `True`、`False`、`None`

### 语句类型

| 语句          | 与Python对比
| :----------- |:--------
| `def`        | 相同。
| `for`        | 迭代序列必须是Tuple/List，部分嵌套场景支持。
| `while`      | 部分嵌套场景支持，对带有while循环的网络求反向不支持。
| `break`      | 相同。
| `if`         | 与Python使用原则一致，但if条件的输入只支持常量。
| `in`         | 仅支持判断常量是否存在于元素都是常量的Tuple/List/Dictionary里。
| `not in`     | 仅支持Dictionary。
| `is`         | 仅支持`True`、`False`、`None`。
| `is not`     | 仅支持`True`、`False`、`None`。
| 赋值语句       | List和Dictionary的多重下标访问不支持作为左值。

### 系统函数/系统类

| 函数/类       | 与Python对比
| :----------- |:--------
| `len`        | 使用原则与Python一致，返回结果与Python一致，返回int。
| `partial`    | 使用原则与Python一致，返回结果与Python不一致，返回函数。
| `map`        | 使用原则与Python一致，返回结果与Python不一致，返回tuple。
| `zip`        | 使用原则与Python一致，返回结果与Python不一致，返回tuple。
| `range`      | 使用原则与Python一致，返回结果与Python不一致，返回tuple。
| `enumerate`  | 使用原则与Python一致，返回结果与Python不一致，返回tuple。
| `super`      | 使用原则与Python一致，返回结果与Python不一致，返回mindspore自定义的命名空间。
| `isinstance` | 使用原则与Python一致，但第二个入参只能是mindspore定义的类型。

### 函数参数

- 参数默认值：目前不支持默认值设为`Tensor`类型数据，支持`int`、`float`、`bool`、`None`、`str`、`tuple`、`list`、`dict`类型数据。
- 可变参数：支持带可变参数网络的推理和训练。
- 键值对参数：目前不支持带键值对参数的函数求反向。
- 可变键值对参数：目前不支持带可变键值对的函数求反向。

### 操作符

| 运算符        | 支持类型
| :----------- |:--------
| `+`          |标量、`Tensor`、`tuple`、`string`
| `-`          |标量、`Tensor`
| `*`          |标量、`Tensor`
| `/`          |标量、`Tensor`
| `**`         |标量、`Tensor`
| `//`         |标量、`Tensor`
| `%`          |标量、`Tensor`
| `[]`         |操作对象类型支持`list`、`tuple`、`Tensor`，支持多重下标访问作为右值，但不支持多重下标访问作为左值，且索引类型仅当操作对象类型为元素类型为`nn.Cell`的tuple或list的取值操作时支持Tensor(这个操作目前Graph模式下仅GPU后端支持)；Tuple、Tensor类型访问限制见切片操作中的说明。

### 索引操作

索引操作包含`tuple`和`Tensor`的索引操作。下面重点介绍一下`Tensor`的索引取值和赋值操作，取值以`tensor_x[index]`为例，赋值以`tensor_x[index] = u`为例进行详细说明。其中tensor_x是一个`Tensor`，对其进行切片操作；index表示索引，u表示赋予的值，可以是`scalar`或者`Tensor(size=1)`。索引类型如下：

- 切片索引：index为`slice`
    - 取值：`tensor_x[start:stop:step]`，其中Slice(start:stop:step)与Python的语法相同，这里不再赘述。
    - 赋值：`tensor_x[start:stop:step]=u`。

- Ellipsis索引：index为`ellipsis`
    - 取值：`tensor_x[...]`。
    - 赋值：`tensor_x[...]=u`。

- 布尔常量索引：index为`True`，index为`False`暂不支持。
    - 取值：`tensor_x[True]`。
    - 赋值：暂不支持。

- Tensor索引：index为`Tensor`
    - 取值：`tensor_x[index]`，`index`必须是`int32`、`int64`类型的`Tensor`，元素取值范围在`[0, tensor_x.shape[0])`。
    - 赋值：`tensor_x[index]=U`。
        - `tensor_x`的数据类型必须是下面一种： `float16`，`float32`，`int8`，`uint8`。
        - `index`必须是`int32`类型的`Tensor`，元素取值范围在`[0, tensor_x.shape[0])`。
        - `U`可以是`Number`，`Tensor`，只包含`Number`的`Tuple`，只包含`Tensor`的`Tuple`。
            - 单个`Number`和`Tuple`里的每个`Number`必须与`tensor_x`的数据类型属于同一类，即
        当`tensor_x`的数据类型是`uint8`或者`int8`时，`Number`类型应该是`int`；
        当`tensor_x`的数据类型是`float16`或者`float32`时，`Number`类型应该是`float`。
            - 单个`Tensor`和`Tuple`里的每个`Tensor`必须与`tensor_x`的数据类型一致，
        单个`Tensor`时，其`shape`需等于或者可广播为`index.shape + tensor_x.shape[1:]`。
            - 包含`Number`的`Tuple`需满足下面条件：
        `len(Tuple) = (index.shape + tensor_x.shape[1:])[-1]`。
            - 包含`Tensor`的`Tuple`需满足下面条件：
        每个`Tensor`的`shape`一样；
        `(len(Tuple),) + Tensor.shape`等于或者可广播为`index.shape + tensor_x.shape[1:]`。

- None常量索引：index为`None`
    - 取值：`tensor_x[None]`，结果与numpy保持一致。
    - 赋值：暂不支持。

- tuple索引：index为`tuple`
    - tuple元素为slice:
        - 取值：例如`tensor_x[::, :4, 3:0:-1]`。
        - 赋值：例如`tensor_x[::, :4, 3:0:-1]=u`。
    - tuple元素为Number:
        - 取值：例如`tensor_x[2,1]`。
        - 赋值：例如`tensor_x[1,4]=u`。
    - tuple元素为slice和ellipsis混合情况:
        - 取值：例如`tensor_x[..., ::, 1:]`
        - 赋值：例如`tensor_x[..., ::, 1:]=u`
    - 其他情况暂不支持

tuple和list类型的索引取值操作，需要重点介绍一下元素类型为`nn.Cell`的tuple或list的索引取值操作，该操作目前在Graph模式下仅GPU后端支持运行，其语法格式形如`layers[index](*inputs)`，具体示例代码如下：

  ```python
  class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index):
          x = self.layers[index](x)
          return x
  ```

同时该语法有以下几个约束：

- 只支持元素类型为`nn.Cell`的tuple或list的索引取值操作。
- 索引值index的类型为`int32`的Tensor标量，取值范围为`[-n, n)`, 其中`n`为tuple的size，支持的tuple的size的最大值为1000。
- tuple中的每个Cell元素的Construct函数的输入数据的数目，类型和shape要求相同，且Construct函数运行后输出的数据的数目，类型和shape也要求相同。
- tuple中的每个Cell元素，需要在tuple定义之前完成定义。
- 该语法不支持做为if、while、for等控制流的运行分支，如果控制流的控制条件为常量除外。举例说明：
    - 支持的写法：

    ```python
    class Net(nn.Cell):
      def __init__(self, flag=True):
          super(Net, self).__init__()
          self.flag = flag
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index):
          if self.flag:
            x = self.layers[index](x)
          return x
    ```

    - 不支持的写法：

    ```python
    class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index, flag):
          if flag:
            x = self.layers[index](x)
          return x
    ```

tuple也支持切片取值操作, 但不支持切片类型为Tensor类型，支持`tuple_x[start:stop:step]`，其中操作对象为与Python的效果相同，这里不再赘述。

### 不支持的语法

目前在网络构造函数里面暂不支持以下语法：
 `raise`、 `yield`、 `async for`、 `with`、 `async with`、 `assert`、 `import`、 `await`。

## 网络定义约束

### 整网实例类型

- 带[@ms_function](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.html#mindspore.ms_function)装饰器的普通Python函数。
- 继承自[nn.Cell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Cell)的Cell子类。

### 网络输入类型

- 整网的训练数据输入参数只能是Tensor类型。
- 生成的ANF图里面不能包含这几种常量节点：字符串类型常量、带有Tuple嵌套的常量、带有List嵌套的常量。

### 网络图优化

 在ME前端图优化过程中，会将DataClass类型、Dictionary、List、键值对操作转换为Tuple相关操作。

### 网络构造组件

| 类别                   | 内容
| :-----------          |:--------
| `Cell`实例             |[mindspore/nn/*](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html)、自定义[Cell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Cell)。
| `Cell`实例的成员函数     | Cell的construct中可以调用其他类成员函数。
| 函数                   | 自定义Python函数、前文中列举的系统函数。
| dataclass实例          | 使用@dataclass装饰的类。
| Primitive算子          |[mindspore/ops/operations/*](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html)
| Composite算子          |[mindspore/ops/composite/*](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html)
| constexpr生成算子       |使用[@constexpr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.constexpr)生成的值计算算子。

### 其他约束

1. 整网`construct`函数输入的参数以及使用`ms_function`装饰器修饰的函数的参数在图编译过程中会进行泛化，不能作为常量输入传给算子使用。所以，在图模式下，限制入口网络的参数只能是`Tensor`，如下例所示：

    - 错误的写法如下：

        ```python
        class ExpandDimsTest(Cell):
            def __init__(self):
                super(ExpandDimsTest, self).__init__()
                self.expandDims = ops.ExpandDims()
    
            def construct(self, input_x, input_axis):
                return self.expandDims(input_x, input_axis)
        expand_dim = ExpandDimsTest()
        input_x = Tensor(np.random.randn(2,2,2,2).astype(np.float32))
        expand_dim(input_x, 0)
        ```

        在示例中，`ExpandDimsTest`是一个只有单算子的网络，网络的输入有`input_x`和`input_axis`两个。因为`ExpandDims`算子的第二个输入需要是常量，这是因为在图编译过程中推导`ExpandDims`算子输出维度的时候需要用到，而`input_axis`作为网络参数输入会泛化成变量，无法确定其值，从而无法推导算子的输出维度导致图编译失败。所以在图编译阶段需要值推导的输入都应该是常量输入。在API中，这类算子需要常量输入的参数会进行说明，标注"constant input is needed"。

    - 正确的写法是在construct函数里面对算子的常量输入直接填入需要的值或者是一个类的成员变量，如下：

        ```python
        class ExpandDimsTest(Cell):
            def __init__(self, axis):
                super(ExpandDimsTest, self).__init__()
                self.expandDims = ops.ExpandDims()
                self.axis = axis

            def construct(self, input_x):
                return self.expandDims(input_x, self.axis)
        axis = 0
        expand_dim = ExpandDimsTest(axis)
        input_x = Tensor(np.random.randn(2,2,2,2).astype(np.float32))
        expand_dim(input_x)
        ```

2. 不允许修改网络的非`Parameter`类型数据成员。示例如下：

    ```python
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.num = 2
            self.par = Parameter(Tensor(np.ones((2, 3, 4))), name="par")

        def construct(self, x, y):
            return x + y
    ```

    上面所定义的网络里，`self.num`不是一个`Parameter`，不允许被修改，而`self.par`是一个`Parameter`，可以被修改。

3. 当`construct`函数里，使用未定义的类成员时，不会像Python解释器那样抛出`AttributeError`，而是作为`None`处理。示例如下：

    ```python
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return x + self.y
    ```

    上面所定义的网络里，`construct`里使用了并未定义的类成员`self.y`，此时会将`self.y`作为`None`处理。
