# Tensor与Parameter

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/model_train/model_building/tensor_and_parameter.md)

## Tensor

张量（[Tensor](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.Tensor.html)）是MindSpore网络运算中的基本数据结构，其功能类似[Numpy数组（ndarray）](https://numpy.org/doc/stable/user/quickstart.html#the-basics)，MindSpore使用Tensor来表示神经网络中传递的数据。

关于Tensor创建、Tensor运算、Tensor与NumPy转换等操作，请参考[张量 Tensor](https://www.mindspore.cn/tutorials/zh-CN/r2.4.0/beginner/tensor.html)。

### Tensor索引支持

Tensor 支持单层与多层索引取值，赋值以及增强赋值，支持动态图(PyNative)以及静态图(Graph)模式。

#### 索引取值

索引值支持`int`、`bool`、`None`、`ellipsis`、`slice`、`Tensor`、`List`、`Tuple`。

- `int`索引取值

  支持单层和多层`int`索引取值，单层`int`索引取值：`tensor_x[int_index]`，多层`int`索引取值：`tensor_x[int_index0][int_index1]...`。

  `int`索引取值操作的是第零维，索引值小于第零维长度，在取出第零维对应位置数据后，会消除第零维。

  例如，如果对一个`shape`为`(3, 4, 5)`的tensor进行单层`int`索引取值，取得结果的`shape`是`(4, 5)`。

  多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`int`索引取值。

  示例如下：

   ```python
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(2 * 3 * 2).reshape((2, 3, 2)))
   data_single = tensor_x[0]
   data_multi = tensor_x[0][1]
   print('data_single:')
   print(data_single)
   print('data_multi:')
   print(data_multi)
   ```

   ```text
   data_single:
   [[0 1]
    [2 3]
    [4 5]]
   data_multi:
   [2 3]
   ```

- `bool`索引取值

  支持单层和多层`bool`索引取值，单层`bool`索引取值：`tensor_x[True]`，多层`bool`索引取值：`tensor_x[True][True]...`。

  `bool`索引取值操作的是第零维，在取出所有数据后，会在`axis=0`轴上扩展一维，对应`True`/`False`，该维长度分别为1/0。`False`将会在`shape`中引入`0`，因此暂只支持`True`。

  例如，对一个`shape`为`(3, 4, 5)`的tensor进行单层`True`索引取值，取得结果的`shape`是`(1, 3, 4, 5)`。

  多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`bool`索引取值。

  示例如下：

   ```python
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(2 * 3).reshape((2, 3)))
   data_single = tensor_x[True]
   data_multi = tensor_x[True][True]
   print('data_single:')
   print(data_single)
   print('data_multi:')
   print(data_multi)
   ```

   ```text
   data_single:
   [[[0 1 2]
     [3 4 5]]]
   data_multi:
   [[[[0 1 2]
      [3 4 5]]]]
   ```

- `None`索引取值

  `None`索引取值和`True`索引取值一致，可参考`True`索引取值，这里不再赘述。

- `ellipsis`索引取值

  支持单层和多层`ellipsis`索引取值，单层`ellipsis`索引取值：`tensor_x[...]`，多层`ellipsis`索引取值：`tensor_x[...][...]...`。

  `ellipsis`索引取值操作在所有维度上取出所有数据。一般多作为`Tuple`索引的组成元素，`Tuple`索引将于下面介绍。

  例如，对一个`shape`为`(3, 4, 5)`的tensor进行`ellipsis`索引取值，取得结果的`shape`依然是`(3, 4, 5)`。

   ```python
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(2 * 3).reshape((2, 3)))
   data_single = tensor_x[...]
   data_multi = tensor_x[...][...]
   print('data_single:')
   print(data_single)
   print('data_multi:')
   print(data_multi)
   ```

   ```text
  data_single:
  [[0 1 2]
   [3 4 5]]
  data_multi:
  [[0 1 2]
   [3 4 5]]
   ```

- `slice`索引取值

  支持单层和多层`slice`索引取值，单层`slice`索引取值：`tensor_x[slice_index]`，多层`slice`索引取值：`tensor_x[slice_index0][slice_index1]...`。

  `slice`索引取值操作的是第零维，取出第零维所切到位置的元素，`slice`不会降维，即使切到长度为1，区别于`int`索引取值。

  例如，`tensor_x[0:1:1] != tensor_x[0]`，因为`shape_former = (1,) + shape_latter`。

  多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`slice`索引取值。

  `slice`有`start`、`stop`和`step`组成。`start`默认值为0，`stop`默认值为该维长度，`step`默认值为1。

  例如，`tensor_x[:] == tensor_x[0:length:1]`。

  示例如下：

   ```python
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(4 * 2 * 2).reshape((4, 2, 2)))
   data_single = tensor_x[1:4:2]
   data_multi = tensor_x[1:4:2][1:]
   print('data_single:')
   print(data_single)
   print('data_multi:')
   print(data_multi)
   ```

   ```text
  data_single:
  [[[ 4  5]
    [ 6  7]]

   [[12 13]
    [14 15]]]
  data_multi:
  [[[12 13]
    [14 15]]]
   ```

- `Tensor`索引取值

  支持单层和多层`Tensor`索引取值，单层`Tensor`索引取值：`tensor_x[tensor_index]`，多层`Tensor`索引取值：`tensor_x[tensor_index0][tensor_index1]...`。

  `Tensor`索引取值操作的是第零维，取出第零维对应位置的元素。

  索引`Tensor`数据类型支持int型和bool型。

  当数据类型是int型时，可以是(int8, int16, int32, int64)，值必须为非负数，且小于第零维长度。

  `Tensor`索引取值得到结果的`data_shape = tensor_index.shape + tensor_x.shape[1:]`。

  例如，对一个`shape`为`(6, 4, 5)`的tensor通过`shape`为`(2, 3)`的tensor进行索引取值，取得结果的`shape`为`(2, 3, 4, 5)`。

  当数据类型是bool型时，`Tensor`索引取值得到结果的维度是 `tensor_x.ndim - tensor_index.ndim + 1`。

  设 `tensor_index` 中True的数量是 `num_true` ，`tensor_x` 的shape是 `(N0, N1, ... Ni-1, Ni, Ni+1, ..., Nk)`, `tensor_index` 的shape是 `(N0, N1, ... Ni-1)`, 则返回值的shape是 `(num_true, Ni+1, Ni+2, ... , Nk)` 。

  示例如下：

   ```python
   from mindspore import dtype as mstype
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor([1, 2, 3])
   tensor_index = ms.Tensor([True, False, True], dtype=mstype.bool_)
   output = tensor_x[tensor_index]
   print(output)
   ```

   ```text
   [1 3]
   ```

  多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`Tensor`索引取值。

  示例如下：

   ```python
   from mindspore import dtype as mstype
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
   tensor_index0 = ms.Tensor(np.array([[1, 2], [0, 3]]), mstype.int32)
   tensor_index1 = ms.Tensor(np.array([[0, 0]]), mstype.int32)
   data_single = tensor_x[tensor_index0]
   data_multi = tensor_x[tensor_index0][tensor_index1]
   print('data_single:')
   print(data_single)
   print('data_multi:')
   print(data_multi)
   ```

   ```text
  data_single:
  [[[[ 6  7  8]
     [ 9 10 11]]

    [[12 13 14]
     [15 16 17]]]


   [[[ 0  1  2]
     [ 3  4  5]]

    [[18 19 20]
     [21 22 23]]]]
  data_multi:
  [[[[[ 6  7  8]
      [ 9 10 11]]

     [[12 13 14]
      [15 16 17]]]


    [[[ 6  7  8]
      [ 9 10 11]]

     [[12 13 14]
      [15 16 17]]]]]
   ```

- `List`索引取值

  支持单层和多层`List`索引取值，单层`List`索引取值：`tensor_x[list_index]`，多层`List`索引取值：`tensor_x[list_index0][list_index1]...`。

  `List`索引取值操作的是第零维，取出第零维对应位置的元素。

  索引`List`数据类型必须是int、bool或两者混合。若数据类型为int，则取值在[`-dimension_shape`, `dimension_shape-1`]之间；若数据类型为bool， 则限制bool个数为对应维度长度，筛选对应维度上值为`True`的元素；若值为前两者混合，则bool类型的`True/False`将转为int类型的`1/0`。

  `List`索引取值得到结果的`data_shape = list_index.shape + tensor_x.shape[1:]`。

  例如，对一个`shape`为`(6, 4, 5)`的tensor通过`shape`为`(3,)`的tensor进行索引取值，取得结果的`shape`为`(3, 4, 5)`。

  多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`List`索引取值。

  示例如下：

   ```python
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
   list_index0 = [1, 2, 0]
   list_index1 = [True, False, True]
   data_single = tensor_x[list_index0]
   data_multi = tensor_x[list_index0][list_index1]
   print('data_single:')
   print(data_single)
   print('data_multi:')
   print(data_multi)
   ```

   ```text
  data_single:
  [[[ 6  7  8]
    [ 9 10 11]]

   [[12 13 14]
    [15 16 17]]

   [[ 0  1  2]
    [ 3  4  5]]]
  data_multi:
  [[[ 6  7  8]
    [ 9 10 11]]

   [[ 0  1  2]
    [ 3  4  5]]]
   ```

- `Tuple`索引取值

  索引`Tuple`的数据类型可以为`int`、`bool`、`None`、`slice`、`ellipsis`、`Tensor`、`List`、`Tuple`。支持单层和多层`Tuple`索引取值，单层`Tuple`索引取值：`tensor_x[tuple_index]`，多层`Tuple`索引取值：`tensor_x[tuple_index0][tuple_index1]...`。`Tuple`中包含的`List`与`Tuple`包含元素规则与单独的`List`规则相同，其他元素规则与单独元素也相同。

  索引`Tuple`中元素按照最终索引Broadcast规则，分为`Basic Index`、`Advanced Index`两类。`Basic Index`包含`slice`、`ellipsis`、`int`与`None`四种类型，`Advanced Index`包含`bool`、`Tensor`、`List`、`Tuple`等类型。索引过程中，所有的`Advanced Index`将会做Broadcast，若`Advaned Index`连续，最终broadcast shape将插入在第一个`Advanced Index`位置；若不连续，则broadcast shape插入在`0`位置。

  索引里除`None`扩展对应维度，`bool`扩展对应维度后与`Advanced Index`做Broadcast。除`ellipsis`、`bool`、`None`外每个元素操作对应位置维度，即`Tuple`中第0个元素操作第零维，第1个元素操作第一维，以此类推。每个元素的索引规则与该元素类型索引取值规则一致。

  `Tuple`索引里最多只有一个`ellipsis`，`ellipsis`前半部分索引元素从前往后对应`Tensor`第零维往后，后半部分索引元素从后往前对应`Tensor`最后一维往前，其他未指定的维度，代表全取。

  元素里包含的`Tensor`数据类型必须是int型或bool型，int型可以是(int8, int16, int32, int64)，值必须为非负数，且小于第零维长度。

  例如，`tensor_x[0:3, 1, tensor_index] == tensor_x[(0:3, 1, tensor_index)]`，因为`0:3, 1, tensor_index`就是一个`Tuple`。

  多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`Tuple`索引取值。

  示例如下：

   ```python
   from mindspore import dtype as mstype
   import mindspore as ms
   import mindspore.numpy as np
   tensor_x = ms.Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)))
   tensor_index = ms.Tensor(np.array([[1, 2, 1], [0, 3, 2]]), mstype.int32)
   data = tensor_x[1, 0:1, tensor_index]
   print('data:')
   print(data)
   ```

   ```text
  data:
  [[[13]
    [14]
    [13]]

   [[12]
    [15]
    [14]]]
   ```

#### 索引赋值

对于形如: `tensor_x[index] = value`， `index`的类型支持`int`、`bool`、`ellipsis`、`slice`、`None`、`Tensor`、`List`、`Tuple`。

`value`的类型支持`Number`、`Tuple`、`List`和`Tensor`。被赋的值会首先被转换为Tensor，数据类型与原Tensor(`tensor_x`)相符。

当`value`为`Number`时，可以理解为将`tensor_x[index]`索引对应元素都更新为`Number`。

当`value`为数组，即只包含`Number`的`Tuple`、`List`或`Tensor`时，`value.shape`需要可以与`tensor_x[index].shape`做广播，将`value`广播到`tensor_x[index].shape`后，更新`tensor_x[index]`对应的值。

当`value`为`Tuple`或`List`时，若`value`中元素包含`Number`，`Tuple`，`List` 和 `Tensor`等多种类型，该`Tuple` 和 `List` 目前只支持一维。

当`value`为`Tuple`或`List`，且存在`Tensor`时，非`Tensor`的元素会首先被转换为`Tensor`，然后这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照`value`为`Tensor`的规则进行赋值。所有`Tensor`的数据类型必须保持一致。

索引赋值可以理解为对索引到的位置元素按照一定规则进行赋值，所有索引赋值都不会改变原`Tensor`的`shape`。

> 当索引中有多个元素指向原Tensor的同一个位置时，该值的更新受底层算子限制，可能出现随机的情况。因此暂不支持索引中重复对Tensor中一个位置的值反复更新。详情请见:[TensorScatterUpdate 算子介绍](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.TensorScatterUpdate.html)
>
> 当前只支持单层索引(`tensor_x[index] = value`)，多层索引(`tensor_x[index1][index2]... = value`)暂不支持。

- `int`索引赋值

  支持单层`int`索引赋值：`tensor_x[int_index] = u`。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_x[1] = 88.0
   tensor_y[1] = np.array([66, 88, 99]).astype(np.float32)
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   ```

   ```text
  tensor_x:
  [[ 0.  1.  2.]
   [88. 88. 88.]]
  tensor_y:
  [[ 0.  1.  2.]
   [66. 88. 99.]]
   ```

- `bool`索引赋值

  支持单层`bool`索引赋值：`tensor_x[bool_index] = u`。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_z = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_x[True] = 88.0
   tensor_y[True] = np.array([66, 88, 99]).astype(np.float32)
   tensor_z[True] = (66, 88, 99)
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   print('tensor_z:')
   print(tensor_z)
   ```

   ```text
  tensor_x:
  [[88. 88. 88.]
   [88. 88. 88.]]
  tensor_y:
  [[66. 88. 99.]
   [66. 88. 99.]]
  tensor_z:
  [[66. 88. 99.]
   [66. 88. 99.]]
   ```

- `ellipsis`索引赋值

  支持单层`ellipsis`索引赋值，单层`ellipsis`索引赋值：`tensor_x[...] = u`。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_z = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_x[...] = 88.0
   tensor_y[...] = np.array([[22, 44, 55], [22, 44, 55]])
   tensor_z[...] = ([11, 22, 33], [44, 55, 66])
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   print('tensor_z:')
   print(tensor_z)
   ```

   ```text
  tensor_x:
  [[88. 88. 88.]
   [88. 88. 88.]]
  tensor_y:
  [[22. 44. 55.]
   [22. 44. 55.]]
  tensor_z:
  [[11. 22. 33.]
   [44. 55. 66.]]
   ```

- `slice`索引赋值

  支持单层`slice`索引赋值：`tensor_x[slice_index] = u`。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_z = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_k = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_x[0:1] = 88.0
   tensor_y[0:2] = 88.0
   tensor_z[0:2] = np.array([[11, 12, 13], [11, 12, 13]]).astype(np.float32)
   tensor_k[0:2] = ([11, 12, 13], (14, 15, 16))
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   print('tensor_z:')
   print(tensor_z)
   print('tensor_k:')
   print(tensor_k)
   ```

   ```text
  tensor_x:
  [[88. 88. 88.]
   [ 3.  4.  5.]
   [ 6.  7.  8.]]
  tensor_y:
  [[88. 88. 88.]
   [88. 88. 88.]
   [ 6.  7.  8.]]
  tensor_z:
  [[11. 12. 13.]
   [11. 12. 13.]
   [ 6.  7.  8.]]
  tensor_k:
  [[11. 12. 13.]
   [14. 15. 16.]
   [ 6.  7.  8.]]
   ```

- `None`索引赋值

  支持单层`None`索引赋值：`tensor_x[none_index] = u`。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_z = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
   tensor_x[None] = 88.0
   tensor_y[None] = np.array([66, 88, 99]).astype(np.float32)
   tensor_z[None] = (66, 88, 99)
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   print('tensor_z:')
   print(tensor_z)
   ```

   ```text
  tensor_x:
  [[88. 88. 88.]
   [88. 88. 88.]]
  tensor_y:
  [[66. 88. 99.]
   [66. 88. 99.]]
  tensor_z:
  [[66. 88. 99.]
   [66. 88. 99.]]
   ```

- `Tensor`索引赋值

  支持单层`Tensor`索引赋值，即`tensor_x[tensor_index] = u`。

  当前支持索引Tensor为 `int` 型和 `bool` 型。

  `int` 型示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_z = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_index = np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32)
   tensor_x[tensor_index] = 88.0
   tensor_y[tensor_index] = np.array([11.0, 12.0, 13.0]).astype(np.float32)
   tensor_z[tensor_index] = [11, 12, 13]
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   print('tensor_z:')
   print(tensor_z)
   ```

   ```text
  tensor_x:
  [[88. 88. 88.]
   [ 3.  4.  5.]
   [88. 88. 88.]]
  tensor_y:
  [[11. 12. 13.]
   [ 3.  4.  5.]
   [11. 12. 13.]]
  tensor_z:
  [[11. 12. 13.]
   [ 3.  4.  5.]
   [11. 12. 13.]]
   ```

  `bool` 型示例如下：

  [[-1. -1. -1.]
   [ 3.  4.  5.]
   [-1. -1. -1.]]

from mindspore import dtype as mstype
import mindspore as ms
tensor_x = ms.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], mstype.float32)
tensor_index = ms.Tensor([True, False, True], mstype.bool_)
tensor_x[tensor_index] = -1
print(tensor_x)

- `List`索引赋值

  支持单层`List`索引赋值：`tensor_x[list_index] = u`。

  `List`索引赋值和`List`索引取值对索引的支持一致。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_index = np.array([[0, 1], [1, 0]]).astype(np.int32)
   tensor_x[[0, 1]] = 88.0
   tensor_y[[True, False, False]] = np.array([11, 12, 13]).astype(np.float32)
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   ```

   ```text
  tensor_x:
  [[88. 88. 88.]
   [88. 88. 88.]
   [ 6.  7.  8.]]
  tensor_y:
  [[11. 12. 13.]
   [ 3.  4.  5.]
   [ 6.  7.  8.]]
   ```

- `Tuple`索引赋值

  支持单层`Tuple`索引赋值：`tensor_x[tuple_index] = u`。

  `Tuple`索引赋值和`Tuple`索引取值对索引的支持一致，但不支持`Tuple`中包含`None`。

  示例如下：

   ```python
   import mindspore.numpy as np
   tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_z = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
   tensor_index = np.array([0, 1]).astype(np.int32)
   tensor_x[1, 1:3] = 88.0
   tensor_y[1:3, tensor_index] = 88.0
   tensor_z[1:3, tensor_index] = np.array([11, 12]).astype(np.float32)
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   print('tensor_z:')
   print(tensor_z)
   ```

   ```text
  tensor_x:
  [[ 0.  1.  2.]
   [ 3. 88. 88.]
   [ 6.  7.  8.]]
  tensor_y:
  [[ 0.  1.  2.]
   [88. 88.  5.]
   [88. 88.  8.]]
  tensor_z:
  [[ 0.  1.  2.]
   [11. 12.  5.]
   [11. 12.  8.]]
   ```

#### 索引增强赋值

增强索引赋值，支持`+=`、`-=`、`*=`、`/=`、`%=`、`**=`、`//=`七种类型，`index`与`value`的规则约束与索引赋值相同。索引值支持`int`、`bool`、`ellipsis`、`slice`、`None`、`Tensor`、`List`、`Tuple`八种类型，赋值支持`Number`、`Tensor`、`Tuple`、`List`四种类型。  

索引增强赋值可以理解为对索引到的位置元素按照一定规则进行取值，取值所得再与`value`进行操作符运算，最终将运算结果进行赋值，所有索引增强赋值都不会改变原`Tensor`的`shape`。

> 当索引中有多个元素指向原Tensor的同一个位置时，该值的更新受底层算子限制，可能出现随机的情况。因此暂不支持索引中重复对Tensor中一个位置的值反复更新。详情请见:[TensorScatterUpdate 算子介绍](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.TensorScatterUpdate.html)。
>
> 目前索引中包含 `True`、`False` 和 `None`的情况暂不支持。

- 规则与约束

  与索引赋值相比，增加了取值与运算的过程。取值过程中`index`的约束规则与索引取值中`index`相同，支持`int`、`bool`、`Tensor`、`Slice`、`Ellipsis`、`None`、`List`与`Tuple`。上述几种类型的数据中所包含`int`值，需在`[-dim_size, dim_size-1]`闭合区间内。
  运算过程中`value`的约束规则与索引赋值中`value`的约束规则相同，`value`类型需为(`Number`、`Tensor`、`List`、`Tuple`)之一，且`value`类型不是`Number`时， `value`的形状需要可以广播到`tensor_x[index]`的形状。

  示例如下：

   ```python
   import mindspore as ms
   tensor_x = ms.Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
   tensor_y = ms.Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
   tensor_x[[0, 1], 1:3] += 2
   tensor_y[[1], ...] -= [4, 3, 2, 1]
   print('tensor_x:')
   print(tensor_x)
   print('tensor_y:')
   print(tensor_y)
   ```

   ```text
  tensor_x:
  [[ 0.  3.  4.  3.]
   [ 4.  7.  8.  7.]
   [ 8.  9. 10. 11.]]
  tensor_y:
  [[ 0.  1.  2.  3.]
   [ 0.  2.  4.  6.]
   [ 8.  9. 10. 11.]]
   ```

### Tensor视图

Tensor视图（Tensor Views）是指一个Tensor经过[view类算子](#view类算子)的返回值，与该Tensor共享内存数据，避免了数据复制，从而可以进行快速且内存高效的重塑、切片和逐元素操作。

例如，要获取Tensor t的视图，可以用t.view(...)。

```python
from mindspore import Tensor
import numpy as np
t = Tensor(np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32))
b = t.view((3, 2))
# Modifying view Tensor changes base Tensor as well.
b[0][0] = 100
print(t[0][0])
# 100
```

由于视图与其原来的Tensor共享底层数据，因此如果在视图中修改数据，它也会反映在原来的Tensor中。

通常，MindSpore算子操作会返回一个新的Tensor作为输出，例如[add()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.add.html)。
但在视图操作的情况下，输出是输入Tensor的视图，以避免不必要的数据复制。创建视图时不会发生数据移动，视图Tensor只是改变它解析相同数据的方式。
使用Tensor Views可能会使内存储存连续的Tensor产生内存存储非连续的Tensor。用户应格外注意，因为连续性可能会对性能产生隐式影响。[transpose()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.transpose.html)是一个常见的例子。

```python
from mindspore import Tensor
import numpy as np
base = Tensor([[0, 1], [2, 3]])
base.is_contiguous()
# True
t = base.transpose(1, 0) # t is a view of base. No data movement happened here.
t.is_contiguous()
# False
# To get a contiguous Tensor, call `.contiguous()` to enforce
# copying data when `t` is not contiguous.
c = t.contiguous()
c.is_contiguous()
# True
```

#### view类算子

作为参考，以下是MindSpore中支持view特性算子的完整列表：

[broadcast_to()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.broadcast_to.html)

[diagonal()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.diagonal.html)

[expand_as()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/Tensor/mindspore.Tensor.expand_as.html)

[movedim()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.movedim.html)

[narrow()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.narrow.html)

[permute()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.permute.html)

[squeeze()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.squeeze.html)

[transpose()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.transpose.html)

[t()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.t.html)

[T](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/Tensor/mindspore.Tensor.T.html)

[unsqueeze()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.unsqueeze.html)

[view()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/Tensor/mindspore.Tensor.view.html)

[view_as()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/Tensor/mindspore.Tensor.view_as.html)

[unbind()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.unbind.html)

[split()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.split.html)

[hsplit()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.hsplit.html)

[vsplit()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.vsplit.html)

[tensor_split()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.tensor_split.html)

[swapaxes()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.swapaxes.html)

[swapdims()](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.swapdims.html)"

## Parameter

参数（[Parameter](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.Parameter.html)）是一类特殊的Tensor，是指在模型训练过程中可以对其值进行更新的变量。MindSpore提供`mindspore.Parameter`类进行Parameter的构造。为了对不同用途的Parameter进行区分，下面对两种不同类别的Parameter进行定义：

- 可训练参数。在模型训练过程中根据反向传播算法求得梯度后进行更新的Tensor，此时需要将`requires_grad`设置为`True`。
- 不可训练参数。不参与反向传播，但需要更新值的Tensor（如BatchNorm中的`mean`和`var`变量），此时需要将`requires_grad`设置为`False`。

> Parameter默认设置`requires_grad=True`。

下面我们构造一个简单的全连接层：

```python
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w') # weight
        self.b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b') # bias

    def construct(self, x):
        z = ops.matmul(x, self.w) + self.b
        return z

net = Network()
```

在`Cell`的`__init__`方法中，我们定义了`w`和`b`两个Parameter，并配置`name`进行命名空间管理。在`construct`方法中使用`self.attr`直接调用参与Tensor运算。

### 获取Parameter

在使用Cell和Parameter构造神经网络层后，我们可以使用多种方法来获取Cell管理的Parameter。

#### 获取单个参数

单独获取某个特定参数，直接调用Python类的成员变量即可。

```python
print(net.b.asnumpy())
```

```text
[-1.2192779  -0.36789745  0.0946381 ]
```

#### 获取可训练参数

可使用`Cell.trainable_params`方法获取可训练参数，通常在配置优化器时需调用此接口。

```python
print(net.trainable_params())
```

```text
[Parameter (name=w, shape=(5, 3), dtype=Float32, requires_grad=True), Parameter (name=b, shape=(3,), dtype=Float32, requires_grad=True)]
```

#### 获取所有参数

使用`Cell.get_parameters()`方法可获取所有参数，此时会返回一个Python迭代器。

```python
print(type(net.get_parameters()))
```

```text
<class 'generator'>
```

或者可以调用`Cell.parameters_and_names`返回参数名称及参数。

```python
for name, param in net.parameters_and_names():
    print(f"{name}:\n{param.asnumpy()}")
```

```text
w:
[[ 4.15680408e-02 -1.20311625e-01  5.02573885e-02]
 [ 1.22175144e-04 -1.34980649e-01  1.17642188e+00]
 [ 7.57667869e-02 -1.74758151e-01 -5.19092619e-01]
 [-1.67846107e+00  3.27240258e-01 -2.06452996e-01]
 [ 5.72323874e-02 -8.27963874e-02  5.94243526e-01]]
b:
[-1.2192779  -0.36789745  0.0946381 ]
```

### 修改Parameter

#### 直接修改参数值

Parameter是一种特殊的Tensor，因此可以使用Tensor索引修改的方式对其值进行修改。

```python
net.b[0] = 1.
print(net.b.asnumpy())
```

```text
[ 1.         -0.36789745  0.0946381 ]
```

#### 覆盖修改参数值

可调用`Parameter.set_data`方法，使用相同Shape的Tensor对Parameter进行覆盖。该方法常用于使用Initializer进行[Cell遍历初始化](https://www.mindspore.cn/docs/zh-CN/r2.4.0/model_train/custom_program/initializer.html#cell%E9%81%8D%E5%8E%86%E5%88%9D%E5%A7%8B%E5%8C%96)。

```python
net.b.set_data(Tensor([3, 4, 5]))
print(net.b.asnumpy())
```

```text
[3. 4. 5.]
```

#### 运行时修改参数值

在深度学习模型训练中，参数的核心功能在于其值的迭代更新，从而优化模型性能。鉴于MindSpore[使用静态图加速](https://www.mindspore.cn/tutorials/zh-CN/r2.4.0/beginner/accelerate_with_static_graph.html)的编译设计，需要使用`mindspore.ops.assign`接口对参数进行赋值。该方法常用于[自定义优化器](https://www.mindspore.cn/docs/zh-CN/r2.4.0/model_train/custom_program/optimizer.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E4%BC%98%E5%8C%96%E5%99%A8)场景。下面是一个简单的运行时修改参数值样例：

```python
import mindspore as ms

@ms.jit
def modify_parameter():
    b_hat = ms.Tensor([7, 8, 9])
    ops.assign(net.b, b_hat)
    return True

modify_parameter()
print(net.b.asnumpy())
```

```text
[7. 8. 9.]
```

### Parameter Tuple

变量元组 `ParameterTuple` 用于保存多个Parameter，该数据类型继承于元组tuple，并提供了克隆功能。

如下示例提供 `ParameterTuple` 创建和克隆方法：

```python
from mindspore.common.initializer import initializer
from mindspore import ParameterTuple
# 创建ParameterTuple
x = Parameter(default_input=ms.Tensor(np.arange(2 * 3).reshape((2, 3))), name="x")
y = Parameter(default_input=initializer('ones', [1, 2, 3], ms.float32), name='y')
z = Parameter(default_input=2.0, name='z')
params = ParameterTuple((x, y, z))

# 克隆ParameterTuple
params_copy = params.clone("params_copy")

print(params)
print(params_copy)
```

```text
  (Parameter (name=x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=z, shape=(), dtype=Float32, requires_grad=True))
  (Parameter (name=params_copy.x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=params_copy.y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=params_copy.z, shape=(), dtype=Float32, requires_grad=True))
```
