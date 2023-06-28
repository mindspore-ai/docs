# 静态图语法支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/note/source_zh_cn/static_graph_syntax_support.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

  在Graph模式下，Python代码并不是由Python解释器去执行，而是将代码编译成静态计算图，然后执行静态计算图。

  关于Graph模式和计算图，可参考文档：<https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/debug_in_pynative_mode.html>

  当前仅支持编译`@ms_function`装饰器修饰的函数、Cell及其子类的实例。
  对于函数，则编译函数定义；对于网络，则编译`construct`方法及其调用的其他方法或者函数。

  `ms_function`使用规则可参考文档：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.ms_function>

  `Cell`定义可参考文档：<https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/cell.html>

  由于语法解析的限制，当前在编译构图时，支持的数据类型、语法以及相关操作并没有完全与Python语法保持一致，部分使用受限。

  本文主要介绍，在编译静态图时，支持的数据类型、语法以及相关操作，这些规则仅适用于Graph模式。

> 以下所有示例都运行在Graph模式下的网络中，为了简洁，并未将网络的定义都写出来。

## 数据类型

### Python内置数据类型

当前支持的`Python`内置数据类型包括：`Number`、`String`、`List`、`Tuple`和`Dictionary`。

#### Number

支持`int`、`float`、`bool`，不支持complex（复数）。

支持在网络里定义`Number`，即支持语法：`y = 1`、`y = 1.2`、 `y = True`。

不支持在网络里强转`Number`，即不支持语法：`y = int(x)`、`y = float(x)`、`y = bool(x)`。

#### String

支持在网络里构造`String`，即支持语法`y = "abcd"`。

不支持在网络里强转`String`，即不支持语法 `y = str(x)`。

#### List

支持在网络里构造`List`，即支持语法`y = [1, 2, 3]`。

不支持在网络里强转`List`，即不支持语法`y = list(x)`。

计算图中最终需要输出的`List`会转换为`Tuple`输出。

- 支持接口

  `append`: 向`list`里追加元素。

  示例如下：

  ```python
  x = [1, 2, 3]
  x.append(4)
  ```

  结果如下：

  ```text
  x: (1, 2, 3, 4)
  ```

- 支持索引取值和赋值

  支持单层和多层索引取值以及赋值。

  取值和赋值的索引值仅支持`int`。

  赋值时，所赋的值支持`Number`、`String`、`Tuple`、`List`、`Tensor`。

  示例如下：

  ```python
  x = [[1, 2], 2, 3, 4]

  y = x[0][1]
  x[1] = Tensor(np.array([1, 2, 3]))
  x[2] = "ok"
  x[3] = (1, 2, 3)
  x[0][1] = 88
  ```

  结果如下：

  ```text
  y: 2
  x: ([1, 88], Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), 'ok', (1, 2, 3))
  ```

#### Tuple

支持在网络里构造`Tuple`，即支持语法`y = (1, 2, 3)`。

不支持在网络里强转`Tuple`，即不支持语法`y = tuple(x)`。

- 支持索引取值

  索引值支持`int`、`slice`、`Tensor`，也支持多层索引取值，即支持语法`data = tuple_x[index0][index1]...`。

  索引值为`Tensor`有如下限制：

    - `tuple`里存放的都是`Cell`，每个`Cell`要在tuple定义之前完成定义，每个`Cell`的入参个数、入参类型和入参`shape`要求一致，每个`Cell`的输出个数、输出类型和输出`shape`也要求一致。

    - 索引`Tensor`是一个`dtype`为`int32`的标量`Tensor`，取值范围在`[-tuple_len, tuple_len)`。

    - 该语法不支持`if`、`while`、`for`控制流条件为变量的运行分支，仅支持控制流条件为常量。

    - 仅支持`GPU`后端。

  `int`、`slice`索引示例如下：

  ```python
  x = (1, (2, 3, 4), 3, 4, Tensor(np.array([1, 2, 3])))
  y = x[1][1]
  z = x[4]
  m = x[1:4]
  ```

  结果如下：

  ```text
  y: 3
  z: Tensor(shape=[3], dtype=Int64, value=[1, 2, 3])
  m: (2, 3, 4), 3, 4)
  ```

  `Tensor`索引示例如下：

  ```python
  class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index):
          ret = self.layers[index](x)
          return ret
  ```

#### Dictionary

支持在网络里构造`Dictionary`，即支持语法`y = {"a": 1, "b": 2}`，当前仅支持`String`作为`key`值。

计算图中最终需要输出的`Dictionary`，会取出所有的`value`组成`Tuple`输出。

- 支持接口

  `keys`：取出`dict`里所有的`key`值，组成`Tuple`返回。

  `values`：取出`dict`里所有的`value`值，组成`Tuple`返回。

  示例如下：

  ```python
  x = {"a": Tensor(np.array([1, 2, 3])), "b": Tensor(np.array([4, 5, 6])), "c": Tensor(np.array([7, 8, 9]))}
  y = x.keys()
  z = x.values()
  ```

  结果如下：

  ```text
  y: ("a", "b", "c")
  z: (Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]), Tensor(shape=[3], dtype=Int64, value=[7, 8, 9]))
  ```

- 支持索引取值和赋值

  取值和赋值的索引值都仅支持`String`。赋值时，所赋的值支持`Number`、`Tuple`、`Tensor`。

  示例如下：

  ```python
  x = {"a": Tensor(np.array([1, 2, 3])), "b": Tensor(np.array([4, 5, 6])), "c": Tensor(np.array([7, 8, 9]))}
  y = x["b"]
  x["a"] = (2, 3, 4)
  ```

  结果如下：

  ```text
  y: Tensor(shape=[3], dtype=Int64, value=[4, 5, 6])
  x: {"a": (2, 3, 4), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]), Tensor(shape=[3], dtype=Int64, value=[7, 8, 9])}
  ```

### MindSpore自定义数据类型

当前MindSpore自定义数据类型包括：`Tensor`、`Primitive`和`Cell`。

#### Tensor

当前不支持在网络里构造Tensor，即不支持语法`x = Tensor(args...)`。

可以通过`@constexpr`装饰器修饰函数，在函数里生成`Tensor`。

关于`@constexpr`的用法可参考：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/ops/mindspore.ops.constexpr.html>

对于网络中需要用到的常量`Tensor`，可以作为网络的属性，在`init`的时候定义，即`self.x = Tensor(args...)`，然后在`construct`里使用。

如下示例，通过`@constexpr`生成一个`shape = (3, 4), dtype = int64`的`Tensor`。

```python
@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4)))
```

下面将介绍下`Tensor`支持的属性、接口、索引取值和索引赋值。

- 支持属性：

  `shape`：获取`Tensor`的shape，返回一个`Tuple`。

  `dtype`：获取`Tensor`的数据类型，返回一个`MindSpore`定义的数据类型。

- 支持接口：

  `all`：对`Tensor`通过`all`操作进行归约，仅支持`Bool`类型的`Tensor`。

  `any`：对`Tensor`通过`any`操作进行归约，仅支持`Bool`类型的`Tensor`。

  `view`：将`Tensor`reshape成输入的`shape`。

  `expand_as`：将`Tensor`按照广播规则扩展成与另一个`Tensor`相同的`shape`。

  示例如下：

  ```python
  x = Tensor(np.array([[True, False, True], [False, True, False]]))
  x_shape = x.shape
  x_dtype = x.dtype
  x_all = x.all()
  x_any = x.any()
  x_view = x.view((1, 6))

  y = Tensor(np.ones((2, 3), np.float32))
  z = Tensor(np.ones((2, 2, 3)))
  y_as_z = y.expand_as(z)
  ```

  结果如下:

  ```text
  x_shape: (2, 3)
  x_dtype: Bool
  x_all: Tensor(shape=[], dtype=Bool, value=False)
  x_any: Tensor(shape=[], dtype=Bool, value=True)
  x_view: Tensor(shape=[1, 6], dtype=Bool, value=[[True, False, True, False, True, False]])

  y_as_z: Tensor(shape=[2, 2, 3], dtype=Float32, value=[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
  ```

- 索引取值

  索引值支持`int`、`True`、`None`、`slice`、`Tensor`、`Tuple`。

    - `int`索引取值

      支持单层和多层`int`索引取值，单层`int`索引取值：`tensor_x[int_index]`，多层`int`索引取值：`tensor_x[int_index0][int_index1]...`。

      `int`索引取值操作的是第0维，索引值小于第0维长度，在取出第0维对应位置数据后，会消除第0维。

      例如，如果对一个`shape`为`(3, 4, 5)`的tensor进行单层`int`索引取值，取得结果的`shape`是`(4, 5)`。

      多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`int`索引取值。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(2 * 3 * 2).reshape((2, 3, 2)))
      data_single = tensor_x[0]
      data_multi = tensor_x[0][1]
      ```

      结果如下：

      ```text
      data_single: Tensor(shape=[3, 2], dtype=Int64, value=[[0, 1], [2, 3], [4, 5]])
      data_multi: Tensor(shape=[2], dtype=Int64, value=[2, 3])
      ```

    - `True`索引取值

      支持单层和多层`True`索引取值，单层`True`索引取值：`tensor_x[True]`，多层`True`索引取值：`tensor_x[True][True]...`。

      `True`索引取值操作的是第0维，在取出所有数据后，会在`axis=0`轴上扩展一维，该维长度为1。

      例如，对一个`shape`为`(3, 4, 5)`的tensor进行单层`True`索引取值，取得结果的`shape`是`(1, 3, 4, 5)`。

      多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`True`索引取值。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(2 * 3 ).reshape((2, 3)))
      data_single = tensor_x[True]
      data_multi = tensor_x[True][True]
      ```

      结果如下：

      ```text
      data_single: Tensor(shape=[1, 2, 3], dtype=Int64, value=[[[0, 1, 2], [3, 4, 5]]])
      data_multi: Tensor(shape=[1, 1, 2, 3], dtype=Int64, value=[[[[0, 1, 2], [3, 4, 5]]]])
      ```

    - `None`索引取值

      `None`索引取值和`True`索引取值一致，可参考`True`索引取值，这里不再赘述。

    - `ellipsis`索引取值

      支持单层和多层`ellipsis`索引取值，单层`ellipsis`索引取值：`tensor_x[...]`，多层`ellipsis`索引取值：`tensor_x[...][...]...`。

      `ellipsis`索引取值操作的是所有维度，原样不动取出所有数据。一般多作为`Tuple`索引的组成元素，`Tuple`索引将于下面介绍。

      例如，对一个`shape`为`(3, 4, 5)`的tensor进行`ellipsis`索引取值，取得结果的`shape`依然是`(3, 4, 5)`。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(2 * 3 ).reshape((2, 3)))
      data_single = tensor_x[...]
      data_multi = tensor_x[...][...]
      ```

      结果如下：

      ```text
      data_single: Tensor(shape=[2, 3], dtype=Int64, value=[[0, 1, 2], [3, 4, 5]])
      data_multi: Tensor(shape=[2, 3], dtype=Int64, value=[[0, 1, 2], [3, 4, 5]])
      ```

    - `slice`索引取值

      支持单层和多层`slice`索引取值，单层`slice`索引取值：`tensor_x[slice_index]`，多层`slice`索引取值：`tensor_x[slice_index0][slice_index1]...`。

      `slice`索引取值操作的是第0维，取出第0维所切到位置的元素，`slice`不会降维，即使切到长度为1，区别于`int`索引取值。

      例如，`tensor_x[0:1:1] != tensor_x[0]`，因为`shape_former = (1,) + shape_latter`。

      多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`slice`索引取值。

      `slice`有`start`、`stop`和`step`组成。`start`默认值为0，`stop`默认值为该维长度，`step`默认值为1。

      例如，`tensor_x[:] == tensor_x[0:length:1]`。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(4 * 2 * 2).reshape((4, 2, 2)))
      data_single = tensor_x[1:4:2]
      data_multi = tensor_x[1:4:2][1:]
      ```

      结果如下：

      ```text
      data_single: Tensor(shape=[2, 2, 2], dtype=Int64, value=[[[4, 5], [6, 7]], [[12, 13], [14, 15]]])
      data_multi: Tensor(shape=[1, 2, 2], dtype=Int64, value=[[[12, 13], [14, 15]]])
      ```

    - `Tensor`索引取值

      支持单层和多层`Tensor`索引取值，单层`Tensor`索引取值：`tensor_x[tensor_index]`，多层`Tensor`索引取值：`tensor_x[tensor_index0][tensor_index1]...`。

      `Tensor`索引取值操作的是第0维，取出第0维对应位置的元素。

      索引`Tensor`数据类型必须是int32，元素不能是负数，值小于第0维长度。

      `Tensor`索引取值得到结果的`data_shape = tensor_index.shape + tensor_x.shape[1:]`。

      例如，对一个`shape`为`(6, 4, 5)`的tensor通过`shape`为`(2, 3)`的tensor进行索引取值，取得结果的`shape`为`(2, 3, 4, 5)`。

      多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`Tensor`索引取值。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
      tensor_index0 = Tensor(np.array([[1, 2], [0, 3]]), mstype.int32)
      tensor_index1 = Tensor(np.array([[0, 0]]), mstype.int32)
      data_single = tensor_x[tensor_index0]
      data_multi = tensor_x[tensor_index0][tensor_index1]
      ```

      结果如下：

      ```text
      data_single: Tensor(shape=[2, 2, 2, 3], dtype=Int64, value=[[[[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[0, 1], [2, 3]], [[12, 13], [14, 15]]]])
      data_multi: Tensor(shape=[1, 2, 2, 2, 3], dtype=Int64, value=[[[[[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[4, 5], [6, 7]], [[8, 9], [10, 11]]]]])
      ```

    - `Tuple`索引取值

      索引`Tuple`的数据类型必须是int32，支持单层和多层`Tuple`索引取值，单层`Tuple`索引取值：`tensor_x[tuple_index]`，多层`Tuple`索引取值：`tensor_x[tuple_index0][tuple_index1]...`。

      `Tuple`索引里的元素可以包含`int`、`slice`、`ellipsis`、`Tensor`。

      索引里除`ellipsis`外每个元素操作对应位置维度，即`Tuple`中第0个元素操作第0维，第1个元素操作第1维，以此类推，每个元素的索引规则与该元素类型索引取值规则一致。

      `Tuple`索引里最多只有一个`ellipsis`，`ellipsis`前半部分索引元素从前往后对应`Tensor`第0维往后，后半部分索引元素从后往前对应`Tensor`最后一维往前，其他未指定的维度，代表全取。

      元素里包含的`Tensor`数据类型必须是int32，且`Tensor`元素不能是负数，值小于操作维度长度。

      例如，`tensor_x[0:3, 1, tensor_index] == tensor_x[(0:3, 1, tensor_index)]`，因为`0:3, 1, tensor_index`就是一个`Tuple`。

      多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`Tuple`索引取值。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)))
      tensor_index = Tensor(np.array([[1, 2, 1], [0, 3, 2]]), mstype.int32)
      data = tensor_x[1, 0:1, tensor_index]
      ```

      结果如下：

      ```text
      data: Tensor(shape=[2, 3, 1], dtype=Int64, value=[[[13], [14], [13]], [[12], [15], [14]]])
      ```

- 索引赋值

  索引值支持`int`、`ellipsis`、`slice`、`Tensor`、`Tuple`。

  索引赋值可以理解为对索引到的位置元素按照一定规则进行赋值，所有索引赋值都不会改变原`Tensor`的`shape`。

  同时支持增强索引赋值，即支持`+=`、`-=`、`*=`、`/=`、`%=`、`**=`、`//=`。

    - `int`索引赋值

      支持单层和多层`int`索引赋值，单层`int`索引赋值：`tensor_x[int_index] = u`，多层`int`索引赋值：`tensor_x[int_index0][int_index1]... = u`。

      所赋值支持`Number`和`Tensor`，`Number`和`Tensor`都会被转为与被更新`Tensor`数据类型一致。

      当所赋值为`Number`时，可以理解为将`int`索引取到位置元素都更新为`Number`。

      当所赋值为`Tensor`时，`Tensor`的`shape`必须等于或者可广播为`int`索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值`Tensor`元素更新到索引取出结果对应元素的原`Tensor`位置。

      例如，对`shape = (2, 3, 4)`的`Tensor`，通过`int`索引1赋值为100，更新后的`Tensor`shape仍为`(2, 3, 4)`，但第0维位置为1的所有元素，值都更新为100。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
      tensor_y = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
      tensor_z = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
      tensor_x[1] = 88.0
      tensor_y[1][1] = 88.0
      tensor_z[1]= Tensor(np.array([66, 88, 99]).astype(np.float32))
      ```

      结果如下：

      ```text
      tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 88.0]])
      tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [3.0, 88.0, 5.0]])
      tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
      ```

    - `ellipsis`索引赋值

      支持单层和多层`ellipsis`索引赋值，单层`ellipsis`索引赋值：`tensor_x[...] = u`，多层`ellipsis`索引赋值：`tensor_x[...][...]... = u`。

      所赋值支持`Number`和`Tensor`，`Number`和`Tensor`里的值都会转为与被更新`Tensor`数据类型一致。

      当所赋值为`Number`时，可以理解为将所有元素都更新为`Number`。

      当所赋值为`Tensor`时，`Tensor`里元素个数必须为1或者等于原`Tensor`里元素个数，元素为1时进行广播，个数相等`shape`不一致时进行`reshape`，
      在保证二者`shape`一致后，将赋值`Tensor`元素按照位置逐一更新到原`Tensor`里。

      例如，对`shape = (2, 3, 4)`的`Tensor`，通过`...`索引赋值为100，更新后的`Tensor`shape仍为`(2, 3, 4)`，所有元素都变为100。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
      tensor_y = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
      tensor_z = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
      tensor_x[...] = 88.0
      tensor_y[...] = Tensor(np.array([22, 44, 55, 22, 44, 55]).astype(np.float32))
      ```

      结果如下：

      ```text
      tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
      tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[22.0, 44.0, 55.0], [22.0, 44.0, 55.0]])
      ```

    - `slice`索引赋值

      支持单层和多层`slice`索引赋值，单层`slice`索引赋值：`tensor_x[slice_index] = u`，多层`slice`索引赋值：`tensor_x[slice_index0][slice_index1]... = u`。

      所赋值支持`Number`和`Tensor`，`Number`和`Tensor`里的值都会转为与被更新`Tensor`数据类型一致。

      当所赋值为`Number`时，可以理解为将`slice`索引取到位置元素都更新为`Number`。

      当所赋值为`Tensor`时，`Tensor`里元素个数必须为1或者等于`slice`索引取到`Tensor`里元素个数，元素为1时进行广播，个数相等`shape`不一致时进行`reshape`，
      在保证二者`shape`一致后，将赋值`Tensor`元素按照位置逐一更新到原`Tensor`里。

      例如，对`shape = (2, 3, 4)`的`Tensor`，通过`0:1:1`索引赋值为100，更新后的`Tensor`shape仍为`(2, 3, 4)`，但第0维位置为0的所有元素，值都更新为100。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_y = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_z = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_x[0:1] = 88.0
      tensor_y[0:2][0:2] = 88.0
      tensor_z[0:2] = Tensor(np.array([11, 12, 13, 11, 12, 13]).astype(np.float32))
      ```

      结果如下：

      ```text
      tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
      tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
      tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [11.0, 12.0, 13.0], [6.0, 7.0, 8.0]])
      ```

    - `Tensor`索引赋值

      仅支持单层`Tensor`索引赋值，即`tensor_x[tensor_index] = u`。

      索引`Tensor`支持`int32`和`bool`类型。

      所赋值支持`Number`、`Tuple`和`Tensor`，`Number`、`Tuple`和`Tensor`里的值必须与原`Tensor`数据类型一致。

      当所赋值为`Number`时，可以理解为将`Tensor`索引取到位置元素都更新为`Number`。

      当所赋值为`Tensor`时，`Tensor`的`shape`必须等于或者可广播为索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值`Tensor`元素更新到索引取出结果对应元素的原`Tensor`位置。

      当所赋值为`Tuple`时，`Tuple`里元素只能全是`Number`或者全是`Tensor`。

      当全是`Number`时，`Number`的类型必须与原`Tensor`数据类型是一类，且元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

      当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

      例如，对一个`shape`为`(6, 4, 5)`、`dtype`为`float32`的tensor通过`shape`为`(2, 3)`的tensor进行索引赋值，如果所赋值为`Number`，则`Number`必须是`float`；
      如果所赋值为`Tuple`，则`tuple`里的元素都得是`float`，且个数为5；如果所赋值为`Tensor`，则`Tensor`的`dtype`必须为`float32`，且`shape`可广播为`(2, 3, 4, 5)`。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_y = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_index = Tensor(np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32))
      tensor_x[tensor_index] = 88.0
      tensor_y[tensor_index] = Tensor(np.array([11.0, 12.0, 13.0]).astype(np.float32))
      ```

      结果如下：

      ```text
      tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [88.0, 88.0, 88.0]])
      tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
      ```

    - `Tuple`索引赋值

      支持单层和多层`Tuple`索引赋值，单层`Tuple`索引赋值：`tensor_x[tuple_index] = u`，多层`Tuple`索引赋值：`tensor_x[tuple_index0][tuple_index1]... = u`。

      `Tuple`索引赋值和`Tuple`索引取值对索引的支持一致, 但多层`Tuple`索引赋值不支持`Tuple`里包含`Tensor`。

      所赋值支持`Number`、`Tuple`和`Tensor`，`Number`、`Tuple`和`Tensor`里的值必须与原`Tensor`数据类型一致。

      当所赋值为`Number`时，可以理解为将`Tensor`索引取到位置元素都更新为`Number`。

      当所赋值为`Tensor`时，`Tensor`的`shape`必须等于或者可广播为索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值`Tensor`元素更新到索引取出结果对应元素的原`Tensor`位置。

      当所赋值为`Tuple`时，`Tuple`里元素只能全是`Number`或者全是`Tensor`。

      当全是`Number`时，`Number`的类型必须与原`Tensor`数据类型是一类，且元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

      当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

      示例如下：

      ```python
      tensor_x = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_y = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_z = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
      tensor_index = Tensor(np.array([[0, 1], [1, 0]]).astype(np.int32))
      tensor_x[1, 1:3] = 88.0
      tensor_y[1:3, tensor_index] = 88.0
      tensor_z[1:3, tensor_index] = Tensor(np.array([11, 12]).astype(np.float32))
      ```

      结果如下：

      ```text
      tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [3.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
      tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 5.0], [88.0, 88.0, 8.0]])
      tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [12.0, 11.0, 5.0], [12.0, 11.0, 8.0]])
      ```

#### Primitive

当前支持在网络里构造`Primitive`及其子类的实例，即支持语法`reduce_sum = ReduceSum(True)`。

但在构造时，参数只能通过位置参数方式传入，不支持通过键值对方式传入，即不支持语法`reduce_sum = ReduceSum(keep_dims=True)`。

当前不支持在网络调用`Primitive`及其子类相关属性和接口。

`Primitive`定义可参考文档：<https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/operators.html>

当前已定义的`Primitive`可参考文档：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.ops.html>

#### Cell

当前支持在网络里构造`Cell`及其子类的实例，即支持语法`cell = Cell(args...)`。

但在构造时，参数只能通过位置参数方式传入，不支持通过键值对方式传入，即不支持在语法`cell = Cell(arg_name=value)`。

当前不支持在网络调用`Cell`及其子类相关属性和接口，除非是在`Cell`自己的`contrcut`中通过`self`调用。

`Cell`定义可参考文档：<https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/cell.html>

当前已定义的`Cell`可参考文档：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.nn.html>

## 运算符

算术运算符和赋值运算符支持`Number`和`Tensor`运算，也支持不同`dtype`的`Tensor`运算。

之所以支持，是因为这些运算符会转换成同名算子进行运算，这些算子支持了隐式类型转换。

规则可参考文档：<https://www.mindspore.cn/doc/note/zh-CN/r1.1/operator_list_implicit.html>

### 算术运算符

| 算术运算符       | 支持类型
| :----------- |:--------
| `+`          |`Number` + `Number`、`Tensor` + `Tensor`、`Tensor` + `Number`、`Tuple` + `Tuple`、`String` + `String`。
| `-`          |`Number` - `Number`、`Tensor` - `Tensor`、`Tensor` - `Number`。
| `*`          |`Number` \* `Number`、`Tensor` \* `Tensor`、`Tensor` \* `Number`。
| `/`          |`Number` / `Number`、`Tensor` / `Tensor`、`Tensor` / `Number`。
| `%`          |`Number` % `Number`、`Tensor` % `Tensor`、`Tensor` % `Number`。
| `**`         |`Number` \*\* `Number`、`Tensor` \*\* `Tensor`、`Tensor` \*\* `Number`。
| `//`         |`Number` // `Number`、`Tensor` // `Tensor`、`Tensor` // `Number`。

### 赋值运算符

| 赋值运算符       | 支持类型
| :----------- |:--------
| `=`          |标量、`Tensor`
| `+=`         |`Number` += `Number`、`Tensor` += `Tensor`、`Tensor` += `Number`、`Tuple` += `Tuple`、`String` += `String`。
| `-=`         |`Number` -= `Number`、`Tensor` -= `Tensor`、`Tensor` -= `Number`。
| `*=`         |`Number` \*= `Number`、`Tensor` \*= `Tensor`、`Tensor` \*= `Number`。
| `/=`         |`Number` /= `Number`、`Tensor` /= `Tensor`、`Tensor` /= `Number`。
| `%=`         |`Number` %= `Number`、`Tensor` %= `Tensor`、`Tensor` %= `Number`。
| `**=`        |`Number` \*\*= `Number`、`Tensor` \*\*= `Tensor`、`Tensor` \*\*= `Number`。
| `//=`        |`Number` //= `Number`、`Tensor` //= `Tensor`、`Tensor` //= `Number`。

### 逻辑运算符

| 逻辑运算符       | 支持类型
| :----------- |:--------
| `and`        |`Number` and `Number`、`Tensor` and `Tensor`。
| `or`         |`Number` or `Number`、`Tensor` or`Tensor`。
| `not`        |not `Number`、not `Tensor`、not `tuple`。

### 成员运算符

| 成员运算符       | 支持类型
| :----------- |:--------
| `in`        |`Number` in `tuple`、`String` in `tuple`、`Tensor` in `Tuple`、`Number` in `List`、`String` in `List`、`Tensor` in `List`、`String` in `Dictionary`。
| `not in`    |与`in`相同。

### 身份运算符

| 身份运算符       | 支持类型
| :----------- |:--------
| `is`        |仅支持判断是`None`、 `True`或者`False`。
| `is not`    |仅支持判断不是`None`、 `True`或者`False`。

## 表达式

### 条件控制语句

#### if

使用方式：

- `if (cond): statements...`

- `x = y if (cond) else z`

参数：`cond` -- 支持类型`Number`、`Tuple`、`List`、`String`、`None`、`Tensor`、`Function`，也可以是计算结果类型是其中之一的表达式。

限制：

- 在构图时，如果`if`未能消除，则`if`分支`return`的数据类型和`shape`，与`if`分支外`return`的数据类型和`shape`必须一致。

- 当只有`if`时，`if`分支变量更新后数据类型和`shape`，与更新前数据类型和`shape`必须一致。

- 当即有`if`又有`else`时，`if`分支变量更新后数据类型和`shape`，与`else`分支更新后数据类型和`shape`必须一致。

示例1：

```python
if x > y:
  return m
else:
  return n
```

`if`分支返回的`m`和`else`分支返回的`n`，二者数据类型和`shape`必须一致。

示例2：

```python
if x > y:
  out = m
else:
  out = n
return out
```

`if`分支更新后`out`和`else`分支更新后`out`，二者数据类型和`shape`必须一致。

### 循环语句

#### for

使用方式：`for i in sequence`

示例如下：

```python
z = Tensor(np.ones((2, 3)))
x = (1, 2, 3)
for i in x:
  z += i
return z
```

结果如下：

```text
z: Tensor(shape=[2, 3], dtype=Int64, value=[[7, 7], [7, 7], [7, 7]])
```

参数：`sequence` -- 遍历序列(`Tuple`、`List`)

#### while

使用方式：`while(cond)`

参数：`cond` -- 与`if`一致。

限制：

- 在构图时，如果`while`未能消除，则`while`内`return`的数据类型和`shape`，与`while`外`return`的数据类型和`shape`必须一致。

- `while`内变量更新后数据类型和`shape`，与更新前数据类型和`shape`必须一致。

示例1：

```python
while x > y:
  x += 1
  return m
return n
```

`while`内返回的`m`和`while`外返回的`n`数据类型必须和`shape`一致。

示例2：

```python
out = m
while x > y:
  x += 1
  out = n
return out
```

`while`内，`out`更新后和更新前的数据类型和`shape`必须一致。

### 流程控制语句

当前流程控制语句支持了`break`、`continue`和`pass`。

#### break

可用于`for`和`while`代码块里，用于终止整个循环。

示例如下：

```python
for i in x:
  if i == 2:
      break
  statement_a
statement_b
```

当 `i == 2`时，循环终止，执行`statement_b`。

#### continue

可用于`for`和`while`语句块里，用于终止本轮循环，直接进入下一轮循环。

示例如下：

```python
for i in x:
  if i == 2:
      continue
  statement_a
statement_b
  ```

当 `i == 2`时，本轮循环终止，不会往下执行`statement_a`，进入下一轮循环。

#### pass

不做任何事情，占位语句。

### 函数定义语句

#### def关键字

用于定义函数。

使用方式：

`def function_name(args): statements...`

示例如下：

```python
def number_add(x, y):
  return x + y
ret = number_add(1, 2)
```

结果如下：

```text
ret: 3
```

#### lambda表达式

用于生成函数。

使用方式：`lambda x, y: x + y`

示例如下：

```python
number_add = lambda x, y: x + y
ret = number_add(2, 3)
```

结果如下：

```text
ret: 5
```

## 函数

### Python内置函数

当前支持的Python内置函数包括：`len`、`isinstance`、`partial`、`map`、`range`、`enumerate`、`super`和`pow`。

#### len

功能：求序列的长度。

调用：`len(sequence)`

入参：`sequence` -- `Tuple`、`List`、`Dictionary`或者`Tensor`。

返回值：序列的长度，类型为`int`。当入参是`Tensor`时，返回的是`Tensor`第0维的长度。

示例如下：

```python
x = (2, 3, 4)
y = [2, 3, 4]
d = {"a": 2, "b": 3}
z = Tensor(np.ones((6, 4, 5)))
x_len = len(x)
y_len = len(y)
d_len = len(d)
z_len = len(z)
```

结果如下：

```text
x_len: 3
y_len: 3
d_len: 2
z_len: 6
  ```

#### isinstance

功能：判断对象是否为类的实例。区别于算子`Isinstance`，该算子的第二个入参是MindSpore的`dtype`模块下定义的类型。

调用：`isinstance(obj, type)`

入参：

- `obj` -- 任意支持类型的任意一个实例。

- `type` -- `MindSpore dtype`模块下的一个类型。

返回值：`obj`为`type`的实例，返回`True`，否则返回`False`。

示例如下：

```python
x = (2, 3, 4)
y = [2, 3, 4]
z = Tensor(np.ones((6, 4, 5)))
x_is_tuple = isinstance(x, mstype.tuple_)
y_is_list= isinstance(y, mstype.list_)
z_is_tensor = isinstance(z, mstype.tensor)
```

结果如下：

```text
x_is_tuple: True
y_is_list: True
z_is_tensor: True
  ```

#### partial

功能：偏函数，固定函数入参。

调用：`partial(func, arg, ...)`

入参：

- `func` -- 函数。

- `arg` -- 一个或多个要固定的参数，支持位置参数和键值对传参。

返回值：返回某些入参固定了值的函数。

示例如下：

```python
def add(x, y):
  return x + y

add_ = partial(add, x=2)
m = add_(y=3)
n = add_(y=5)
```

结果如下：

```text
m: 5
n: 7
```

#### map

功能：根据提供的函数对一个或者多个序列做映射，由映射的结果生成一个新的序列。
如果多个序列中的元素个数不一致，则生成的新序列与最短的那个长度相同。

调用：`map(func, sequence, ...)`

入参：

- `func` -- 函数。

- `sequence` -- 一个或多个序列（`Tuple`或者`List`）。

返回值：返回一个`Tuple`。

示例如下：

```python
def add(x, y):
  return x + y

elements_a = (1, 2, 3)
elements_b = (4, 5, 6)
ret = map(add, elements_a, elements_b)
```

结果如下：

```text
ret: (5, 7, 9)
```

#### zip

功能：将多个序列中对应位置的元素打包成一个个元组，然后由这些元组组成一个新序列，
如果各个序列中的元素个数不一致，则生成的新序列与最短的那个长度相同。

调用：`zip(sequence, ...)`

入参：`sequence` -- 一个或多个序列(`Tuple`或`List`)`。

返回值：返回一个`Tuple`。

示例如下：

```python
elements_a = (1, 2, 3)
elements_b = (4, 5, 6)
ret = zip(elements_a, elements_b)
```

结果如下：

```text
ret: ((1, 4), (2, 5), (3, 6))
```

#### range

功能：根据起始值、结束值和步长创建一个`Tuple`。

调用：

- `range(start, stop, step)`

- `range(start, stop)`

- `range(stop)`

入参：

- `start` -- 计数起始值，类型为`int`，默认为0。

- `stop` -- 计数结束值，但不包括在内，类型为`int`。

- `step` -- 步长，类型为`int`，默认为1。

返回值：返回一个`Tuple`。

示例如下：

```python
x = range(0, 6, 2)
y = range(0, 5)
z = range(3)
```

结果如下：

```text
x: (0, 2, 4)
y: (0, 1, 2, 3, 4)
z: (0, 1, 2)
```

#### enumerate

功能：生成一个序列的索引序列，索引序列包含数据和对应下标。

调用：

- `enumerate(sequence, start)`

- `enumerate(sequence)`

入参：

- `sequence` -- 一个序列（`Tuple`、`List`、`Tensor`）。

- `start` -- 下标起始位置，类型为`int`，默认为0。

返回值：返回一个`Tuple`。

示例如下：

```python
x = (100, 200, 300, 400)
y = Tensor(np.array([[1, 2], [3, 4], [5 ,6]]))
m = enumerate(x, 3)
n = enumerate(y)
```

结果如下：

```text
m: ((3, 100), (4, 200), (5, 300), (5, 400))
n: ((0, Tensor(shape=[2], dtype=Int64, value=[1, 2])), (1, Tensor(shape=[2], dtype=Int64, value=[3, 4])), (2, Tensor(shape=[2], dtype=Int64, value=[5, 6])))
```

#### super

功能：用于调用父类(超类)的一个方法，一般在`super`之后调用父类的方法。

调用：

- `super().xxx()`

- `super(type, self).xxx()`

入参：

- `type` -- 类。

- `self` -- 对象。

返回值：返回父类的方法。

示例如下：

```python
class FatherNet(nn.Cell):
  def __init__(self, x):
      super(FatherNet, self).__init__(x)
      self.x = x

  def construct(self, x, y):
      return self.x * x

  def test_father(self, x):
      return self.x + x

class SingleSubNet(FatherNet):
def __init__(self, x, z):
    super(SingleSubNet, self).__init__(x)
    self.z = z

def construct(self, x, y):
    ret_father_construct = super().construct(x, y)
    ret_father_test = super(SingleSubNet, self).test_father(x)
    return ret_father_construct, ret_father_test
```

#### pow

功能：求幂。

调用：`pow(x, y)`

入参：

- `x` -- 底数， `Number`或`Tensor`。

- `y` -- 幂指数， `Number`或`Tensor`。

返回值：返回`x`的`y`次幂，`Number`或`Tensor`。

示例如下：

```python
x = Tensor(np.array([1, 2, 3]))
y = Tensor(np.array([1, 2, 3]))
ret = pow(x, y)
```

结果如下：

```text
ret: Tensor(shape=[3], dtype=Int64, value=[1, 4, 27]))
```

#### print

功能：用于打印。

调用：`print(arg, ...)`

入参：`arg` -- 要打印的信息(`String`或`Tensor`）。

返回值：无返回值。

示例如下：

```python
x = Tensor(np.array([1, 2, 3]))
print("result", x)
```

结果如下：

```text
result Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]))
```

### 函数参数

- 参数默认值：目前不支持默认值设为`Tensor`类型数据，支持`int`、`float`、`bool`、`None`、`str`、`tuple`、`list`、`dict`类型数据。

- 可变参数：支持带可变参数网络的推理和训练。

- 键值对参数：目前不支持带键值对参数的函数求反向。

- 可变键值对参数：目前不支持带可变键值对的函数求反向。

## 网络定义

### 整网实例类型

- 带[@ms_function](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.ms_function)装饰器的普通Python函数。

- 继承自[nn.Cell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.Cell.html)的Cell子类。

### 网络构造组件

| 类别                     | 内容
| :-----------             |:--------
| `Cell`实例               |[mindspore/nn/*](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.nn.html)、自定义[Cell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.Cell.html)。
| `Cell`实例的成员函数       | Cell的construct中可以调用其他类成员函数。
| `dataclass`实例          | 使用@dataclass装饰的类。
| `Primitive`算子          |[mindspore/ops/operations/*](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.ops.html)
| `Composite`算子          |[mindspore/ops/composite/*](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.ops.html)
| `constexpr`生成算子       |使用[@constexpr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/ops/mindspore.ops.constexpr.html)生成的值计算算子。
| 函数                     | 自定义Python函数、前文中列举的系统函数。

### 网络使用约束

1. 当前整网入参（即最外层网络入参）默认仅支持`Tensor`，如果要支持非`Tensor`，可设置网络的`support_non_tensor_inputs`属性为`True`。

   在网络初始化的时候，设置`self.support_non_tensor_inputs = True`，该配置目前仅支持正向网络，暂不支持反向网络，即不支持对整网入参有非`Tensor`的网络求反向。

   支持最外层传入标量示例如下：

   ```python
   class ExpandDimsNet(nn.Cell):
       def __init__(self):
           super(ExpandDimsNet, self).__init__()
           self.support_non_tensor_inputs = True
           self.expandDims = ops.ExpandDims()

       def construct(self, input_x, input_axis):
           return self.expandDims(input_x, input_axis)
   expand_dim_net = ExpandDimsNet()
   input_x = Tensor(np.random.randn(2,2,2,2).astype(np.float32))
   expand_dim_net(input_x, 0)
   ```

2. 不允许修改网络的非`Parameter`类型数据成员。

   示例如下：

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

3. 当`construct`函数里，使用未定义的类成员时，不会像Python解释器那样抛出`AttributeError`，而是作为`None`处理。

   示例如下：

   ```python
   class Net(Cell):
       def __init__(self):
           super(Net, self).__init__()

       def construct(self, x):
           return x + self.y
    ```

   上面所定义的网络里，`construct`里使用了并未定义的类成员`self.y`，此时会将`self.y`作为`None`处理。
