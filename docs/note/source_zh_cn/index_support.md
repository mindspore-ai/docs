# Tensor 索引支持

`Linux` `Ascend` `GPU` `模型开发` `初级` `中级` `高级`

<!-- TOC -->

- [Tensor 索引支持](#tensor-索引支持)
    - [索引取值](#索引取值)
    - [索引赋值](#索引赋值)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/note/source_zh_cn/index_support.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

Tensor 支持单层与多层索引取值，赋值以及增强赋值，支持动态图(PyNative)以及静态图(Graph)模式。

## 索引取值

索引值支持`int`、`bool`、`None`、`slice`、`Tensor`、`List`、`Tuple`。

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

- `bool`索引取值

    支持单层和多层`bool`索引取值，单层`bool`索引取值：`tensor_x[True]`，多层`bool`索引取值：`tensor_x[True][False]...`。

    `bool`索引取值操作的是第0维，在取出所有数据后，会在`axis=0`轴上扩展一维，对应`True`/`False`, 该维长度分别为1/0。

    例如，对一个`shape`为`(3, 4, 5)`的tensor进行单层`True`索引取值，取得结果的`shape`是`(1, 3, 4, 5)`，对一个`shape`为`(3, 4, 5)`的tensor进行单层`False`索引取值，取得结果的`shape`是`(0, 3, 4, 5)`。

    多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`bool`索引取值。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(2 * 3 ).reshape((2, 3)))
    data_single = tensor_x[True]
    data_multi = tensor_x[True][False]
    ```

    结果如下：

    ```text
    data_single: Tensor(shape=[1, 2, 3], dtype=Int64, value=[[[0, 1, 2], [3, 4, 5]]])
    data_multi: Tensor(shape=[1, 0, 2, 3], dtype=Int64, value=[[[[], []]]])
    ```

- `None`索引取值

    `None`索引取值和`True`索引取值一致，可参考`True`索引取值，这里不再赘述。

- `ellipsis`索引取值

    支持单层和多层`ellipsis`索引取值，单层`ellipsis`索引取值：`tensor_x[...]`，多层`ellipsis`索引取值：`tensor_x[...][...]...`。

    `ellipsis`索引取值操作在所有维度上取出所有数据。一般多作为`Tuple`索引的组成元素，`Tuple`索引将于下面介绍。

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

    索引`Tensor`数据类型必须是int型，可以是(int8, int16, int32, int64)，值必须为非负数，且小于第0维长度。

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

- `List`索引取值

    支持单层和多层`List`索引取值，单层`List`索引取值：`tensor_x[list_index]`，多层`List`索引取值：`tensor_x[list_index0][list_index1]...`。

    `List`索引取值操作的是第0维，取出第0维对应位置的元素。

    索引`List`数据类型必须是int、bool或两者混合。若数据类型为int，则取值在[`-dimension_shape`, `dimension_shape-1`]之间；若数据类型为bool， 则限制bool个数为对应维度长度，筛选对应维度上值为`True`的元素；若值为前两者混合，则bool类型的`True/False`将转为int类型的`1/0`。

    `List`索引取值得到结果的`data_shape = list_index.shape + tensor_x.shape[1:]`。

    例如，对一个`shape`为`(6, 4, 5)`的tensor通过`shape`为`(3,)`的tensor进行索引取值，取得结果的`shape`为`(3, 4, 5)`。

    多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`List`索引取值。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
    list_index0 = [1, 2, 0]
    list_index1 = [True, False, True]
    data_single = tensor_x[list_index0]
    data_multi = tensor_x[list_index0][list_index1]
    ```

    结果如下：

    ```text
    data_single: Tensor(shape=[3, 2, 3], dtype=Int64, value=[[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]], [[0, 1, 2], [3, 4, 5]]])
    data_multi: Tensor(shape=[2, 2, 3], dtype=Int64, value=[[[6, 7, 8], [9, 10, 11]], [[0, 1, 2], [3, 4, 5]]])
    ```

- `Tuple`索引取值

    索引`Tuple`的数据类型可以为`int`、`bool`、`None`、`slice`、`ellipsis`、`Tensor`、`List`、`Tuple`。支持单层和多层`Tuple`索引取值，单层`Tuple`索引取值：`tensor_x[tuple_index]`，多层`Tuple`索引取值：`tensor_x[tuple_index0][tuple_index1]...`。`Tuple`中包含的`List`与`Tuple`包含元素规则与单独的`List`规则相同，其他元素规则与单独元素也相同。

    索引`Tuple`中元素按照最终索引Broadcast规则，分为`Basic Index`、`Advanced Index`两类。`Basic Index`包含`slice`、`ellipsis`与`None`三种类型，`Advanced Index`包含`int`、`bool`、`Tensor`、`List`、`Tuple`等五种类型。索引过程中，所有的`Advanced Index`将会做Broadcast，若`Advaned Index`连续，最终broadcast shape将插入在第一个`Advanced Index`位置；若不连续，则broadcast shape插入在`0`位置。

    索引里除`None`扩展对应维度，`bool`扩展对应维度后与`Advanced Index`做Broadcast。除`ellipsis`、`bool`、`None`外每个元素操作对应位置维度，即`Tuple`中第0个元素操作第0维，第1个元素操作第1维，以此类推。每个元素的索引规则与该元素类型索引取值规则一致。

    `Tuple`索引里最多只有一个`ellipsis`，`ellipsis`前半部分索引元素从前往后对应`Tensor`第0维往后，后半部分索引元素从后往前对应`Tensor`最后一维往前，其他未指定的维度，代表全取。

    元素里包含的`Tensor`数据类型必须是int型，可以是(int8, int16, int32, int64)，值必须为非负数，且小于第0维长度。

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

## 索引赋值

索引值支持`int`、`ellipsis`、`slice`、`Tensor`、`Tuple`。

索引赋值可以理解为对索引到的位置元素按照一定规则进行赋值，所有索引赋值都不会改变原`Tensor`的`shape`。

同时支持增强索引赋值，即支持`+=`、`-=`、`*=`、`/=`、`%=`、`**=`、`//=`。

- `int`索引赋值

    支持单层和多层`int`索引赋值，单层`int`索引赋值：`tensor_x[int_index] = u`，多层`int`索引赋值：`tensor_x[int_index0][int_index1]... = u`。

    所赋值支持`Number`, `Tuple`, `List`, 和`Tensor`，所赋值都会被转为与被更新`Tensor`数据类型一致。

    当所赋值为`Number`时，可以理解为将所有元素都更新为`Number`。

    当所赋值为数组，即`Tuple`, `List`, 或`Tensor`时，数组的`shape`必须等于或者可广播为`int`索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值数组元素更新到索引取出结果对应元素的原`Tensor`位置。

    当所赋值为`Tuple`或`List`时，所赋元素如果存在`Number`, `Tuple`, `List` 和 `Tensor`的混合情况， 该`Tuple` 和 `List` 目前只支持1维。

    当全是`Number`时，元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

    当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

    例如，对`shape = (2, 3, 4)`的`Tensor`，通过`int`索引1赋值为100，更新后的`Tensor`shape仍为`(2, 3, 4)`，但第0维位置为1的所有元素，值都更新为100。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_y = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_z = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_k = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_x[1] = 88.0
    tensor_y[1][1] = 88.0
    tensor_z[1]= Tensor(np.array([66, 88, 99]).astype(np.float32))
    tensor_k[1] = (66, np.array(88), 99)
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [3.0, 88.0, 5.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
    tensor_k: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
    ```

- `ellipsis`索引赋值

    支持单层和多层`ellipsis`索引赋值，单层`ellipsis`索引赋值：`tensor_x[...] = u`，多层`ellipsis`索引赋值：`tensor_x[...][...]... = u`。

    所赋值支持`Number`, `Tuple`, `List`, 和`Tensor`，所赋值都会被转为与被更新`Tensor`数据类型一致。

    当所赋值为`Number`时，可以理解为将所有元素都更新为`Number`。

    当所赋值为数组，即`Tuple`, `List`, 或`Tensor`时，数组里元素个数必须为1或者等于原`Tensor`里元素个数，元素为1时进行广播，个数相等`shape`不一致时进行`reshape`，
    在保证二者`shape`一致后，将赋值`Tensor`元素按照位置更新到原`Tensor`里。

    当所赋值为`Tuple`或`List`时，所赋元素如果存在`Number`, `Tuple`, `List` 和 `Tensor`的混合情况， 该`Tuple` 和 `List` 目前只支持1维。

    当全是`Number`时，元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

    当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

    例如，对`shape = (2, 3, 4)`的`Tensor`，通过`...`索引赋值为100，更新后的`Tensor`shape仍为`(2, 3, 4)`，所有元素都变为100。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_y = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_z = Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float32))
    tensor_x[...] = 88.0
    tensor_y[...] = Tensor(np.array([22, 44, 55, 22, 44, 55]).astype(np.float32))
    tensor_z[...] = ([11, 22, 33], [44, 55, 66])
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[22.0, 44.0, 55.0], [22.0, 44.0, 55.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Int64, value=[[11, 22, 33], [44, 55, 66]])
    ```

- `slice`索引赋值

    支持单层和多层`slice`索引赋值，单层`slice`索引赋值：`tensor_x[slice_index] = u`，多层`slice`索引赋值：`tensor_x[slice_index0][slice_index1]... = u`。

    所赋值支持`Number`, `Tuple`, `List`, 和`Tensor`，所赋值都会被转为与被更新`Tensor`数据类型一致。

    当所赋值为`Number`时，可以理解为将所有元素都更新为`Number`。

    当所赋值为数组，即`Tuple`, `List`, 或`Tensor`时，数组的`shape`必须等于或者可广播为`slice`索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值数组元素更新到索引取出结果对应元素的原`Tensor`位置。

    当所赋值为`Tuple`或`List`时，所赋元素如果存在`Number`, `Tuple`, `List` 和 `Tensor`的混合情况， 该`Tuple` 和 `List` 目前只支持1维。

    当全是`Number`时，元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

    当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

    例如，对`shape = (2, 3, 4)`的`Tensor`，通过`0:1:1`索引赋值为100，更新后的`Tensor`shape仍为`(2, 3, 4)`，但第0维位置为0的所有元素，值都更新为100。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_y = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_z = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_k = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_x[0:1] = 88.0
    tensor_y[0:2][0:2] = 88.0
    tensor_z[0:2] = Tensor(np.array([11, 12, 13, 11, 12, 13]).astype(np.float32))
    tensor_k[0:2] = ([11, 12, 13], (14, 15, 16))
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [11.0, 12.0, 13.0], [6.0, 7.0, 8.0]])
    tensor_k: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [6.0, 7.0, 8.0]])
    ```

- `Tensor`索引赋值

    仅支持单层`Tensor`索引赋值，即`tensor_x[tensor_index] = u`。

    当前不支持索引Tensor为`bool`类型。

    所赋值支持`Number`, `Tuple`, `List`, 和`Tensor`，所赋值都会被转为与被更新`Tensor`数据类型一致。

    当所赋值为`Number`时，可以理解为将所有元素都更新为`Number`。

    当所赋值为数组，即`Tuple`, `List`, 或`Tensor`时，数组的`shape`必须等于或者可广播为`Tensor`索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值数组元素更新到索引取出结果对应元素的原`Tensor`位置。

    当所赋值为`Tuple`或`List`时，所赋元素如果存在`Number`, `Tuple`, `List` 和 `Tensor`的混合情况， 该`Tuple` 和 `List` 目前只支持1维。

    当全是`Number`时，`Number`的类型必须与原`Tensor`数据类型是一类，且元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

    当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

    例如，对一个`shape`为`(6, 4, 5)`、`dtype`为`float32`的tensor通过`shape`为`(2, 3)`的tensor进行索引赋值，如果所赋值为`Number`，则`Number`必须是`float`；
    如果所赋值为`Tuple`，则`tuple`里的元素都得是`float`，且个数为5；如果所赋值为`Tensor`，则`Tensor`的`dtype`必须为`float32`，且`shape`可广播为`(2, 3, 4, 5)`。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_y = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_z = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_index = Tensor(np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32))
    tensor_x[tensor_index] = 88.0
    tensor_y[tensor_index] = Tensor(np.array([11.0, 12.0, 13.0]).astype(np.float32))
    tensor_z[tensor_index] = [11, 12, 13]
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
    ```

- `Tuple`索引赋值

    支持单层和多层`Tuple`索引赋值，单层`Tuple`索引赋值：`tensor_x[tuple_index] = u`，多层`Tuple`索引赋值：`tensor_x[tuple_index0][tuple_index1]... = u`。

    `Tuple`索引赋值和`Tuple`索引取值对索引的支持一致, 但多层`Tuple`索引赋值不支持`Tuple`里包含`Tensor`。

    所赋值支持`Number`, `Tuple`, `List`, 和`Tensor`，所赋值都会被转为与被更新`Tensor`数据类型一致。

    当所赋值为`Number`时，可以理解为将所有元素都更新为`Number`。

    当所赋值为数组，即`Tuple`, `List`, 或`Tensor`时，数组的`shape`必须等于或者可广播为`int`索引取到结果的`shape`，在保持二者`shape`一致后，然后将赋值数组元素更新到索引取出结果对应元素的原`Tensor`位置。

    当所赋值为`Tuple`或`List`时，所赋元素如果存在`Number`, `Tuple`, `List` 和 `Tensor`的混合情况， 该`Tuple` 和 `List` 目前只支持1维。

    当全是`Number`时，`Number`的类型必须与原`Tensor`数据类型是一类，且元素的个数必须等于索引取到结果`shape`的最后一维，然后广播为索引取到结果`shape`；

    当全是`Tensor`的时候，这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照所赋值为`Tensor`的规则进行赋值。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_y = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_z = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_k = Tensor(np.arange(3 * 3).reshape((3, 3)).astype(np.float32))
    tensor_index = Tensor(np.array([[0, 1], [1, 0]]).astype(np.int32))
    tensor_x[1, 1:3] = 88.0
    tensor_y[1:3, tensor_index] = 88.0
    tensor_z[1:3, tensor_index] = Tensor(np.array([11, 12]).astype(np.float32))
    tensor_k[..., [2]] = [6, 6, 6]
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [3.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 5.0], [88.0, 88.0, 8.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [12.0, 11.0, 5.0], [12.0, 11.0, 8.0]])
    tensor_k: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 6.0], [3.0, 4.0, 6.0], [6.0, 7.0, 6.0]])
    ```
