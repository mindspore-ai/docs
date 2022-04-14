# Tensor索引支持

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/index_support.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

Tensor 支持单层与多层索引取值，赋值以及增强赋值，支持动态图(PyNative)以及静态图(Graph)模式。

## 索引取值

索引值支持`int`、`bool`、`None`、`ellipsis`、`slice`、`Tensor`、`List`、`Tuple`。

- `int`索引取值

    支持单层和多层`int`索引取值，单层`int`索引取值：`tensor_x[int_index]`，多层`int`索引取值：`tensor_x[int_index0][int_index1]...`。

    `int`索引取值操作的是第0维，索引值小于第0维长度，在取出第0维对应位置数据后，会消除第0维。

    例如，如果对一个`shape`为`(3, 4, 5)`的tensor进行单层`int`索引取值，取得结果的`shape`是`(4, 5)`。

    多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`int`索引取值。

    示例如下：

    ```python
    from mindspore import Tensor
    import mindspore.numpy as np
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

    支持单层和多层`bool`索引取值，单层`bool`索引取值：`tensor_x[True]`，多层`bool`索引取值：`tensor_x[True][True]...`。

    `bool`索引取值操作的是第0维，在取出所有数据后，会在`axis=0`轴上扩展一维，对应`True`/`False`，该维长度分别为1/0。`False`将会在`shape`中引入`0`，因此暂只支持`True`。

    例如，对一个`shape`为`(3, 4, 5)`的tensor进行单层`True`索引取值，取得结果的`shape`是`(1, 3, 4, 5)`。

    多层索引取值可以理解为，后一层索引取值在前一层索引取值结果上再进行`bool`索引取值。

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
    data_single: Tensor(shape=[2, 2, 2, 3], dtype=Int64, value=[[[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16 ,17]]], [[[0, 1, 2], [3, 4, 5]], [[18, 19, 20], [21, 22, 23]]]])
    data_multi: Tensor(shape=[1, 2, 2, 2, 3], dtype=Int64, value=[[[[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16 ,17]]], [[[0, 1, 2], [3, 4, 5]], [[18, 19, 20], [21, 22, 23]]]]]))
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

对于形如: `tensor_x[index] = value`， `index`的类型支持`int`、`bool`、`ellipsis`、`slice`、`None`、`Tensor`、`List`、`Tuple`。

`value`的类型支持`Number`、`Tuple`、`List`和`Tensor`。被赋的值会首先被转换为张量，数据类型与原张量(`tensor_x`)相符。

当`value`为`Number`时，可以理解为将`tensor_x[index]`索引对应元素都更新为`Number`。

当`value`为数组，即只包含`Number`的`Tuple`、`List`或`Tensor`时，`value.shape`需要可以与`tensor_x[index].shape`做广播，将`value`广播到`tensor_x[index].shape`后，更新`tensor_x[index]`对应的值。

当`value`为`Tuple`或`List`时，若`value`中元素包含`Number`，`Tuple`，`List` 和 `Tensor`等多种类型，该`Tuple` 和 `List` 目前只支持1维。

当`value`为`Tuple`或`List`，且存在`Tensor`时，非`Tensor`的元素会首先被转换为`Tensor`，然后这些`Tensor`在`axis=0`轴上打包之后成为一个新的赋值`Tensor`，这时按照`value`为`Tensor`的规则进行赋值。所有`Tensor`的数据类型必须保持一致。

索引赋值可以理解为对索引到的位置元素按照一定规则进行赋值，所有索引赋值都不会改变原`Tensor`的`shape`。

> 当索引中有多个元素指向原张量的同一个位置时，该值的更新受底层算子限制，可能出现随机的情况。因此暂不支持索引中重复对张量中一个位置的值反复更新。详情请见:[TensorScatterUpdate 算子介绍](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.TensorScatterUpdate.html)
>
> 当前只支持单层索引(`tensor_x[index] = value`)， 多层索引(`tensor_x[index1][index2]... = value`)暂不支持。

- `int`索引赋值

    支持单层`int`索引赋值：`tensor_x[int_index] = u`。

    示例如下：

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_y = np.arange(2 *3).reshape((2, 3)).astype(np.float32)
    tensor_z = np.arange(2* 3).reshape((2, 3)).astype(np.float32)
    tensor_x[1] = 88.0
    tensor_y[1]= np.array([66, 88, 99]).astype(np.float32)
    tensor_z[1] = (66, np.array(88).astype(np.int64), 99)
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
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
    tensor_y[True]= np.array([66, 88, 99]).astype(np.float32)
    tensor_z[True] = (66, 88, 99)
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
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
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[22.0, 44.0, 55.0], [22.0, 44.0, 55.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[11., 22., 33.], [44., 55., 66.]])
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
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [11.0, 12.0, 13.0], [6.0, 7.0, 8.0]])
    tensor_k: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [6.0, 7.0, 8.0]])
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
    tensor_y[None]= np.array([66, 88, 99]).astype(np.float32)
    tensor_z[None] = (66, 88, 99)
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    ```

- `Tensor`索引赋值

    支持单层`Tensor`索引赋值，即`tensor_x[tensor_index] = u`。

    当前不支持索引Tensor为`bool`类型，只能为`mstype.int*`型。

    示例如下：

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_z = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_index = np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32)
    tensor_x[tensor_index] = 88.0
    tensor_y[tensor_index] = np.array([11.0, 12.0, 13.0]).astype(np.float32)
    tensor_z[tensor_index] = [11, 12, 13]
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
    ```

- `List`索引赋值

    支持单层`List`索引赋值：`tensor_x[list_index] = u`。

    `List`索引赋值和`List`索引取值对索引的支持一致。

    示例如下：

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_index = np.array([[0, 1], [1, 0]]).astype(np.int32)
    tensor_x[[0,1]] = 88.0
    tensor_y[[True, False, False]] = np.array([11, 12, 13]).astype(np.float32)
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
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
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [3.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 5.0], [88.0, 88.0, 8.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [11.0, 12.0, 5.0], [11.0, 12.0, 8.0]])
    ```

## 索引增强赋值

增强索引赋值，支持`+=`、`-=`、`*=`、`/=`、`%=`、`**=`、`//=`七种类型，`index`与`value`的规则约束与索引赋值相同。索引值支持`int`、`bool`、`ellipsis`、`slice`、`None`、`Tensor`、`List`、`Tuple`八种类型，赋值支持`Number`、`Tensor`、`Tuple`、`List`四种类型。  

索引增强赋值可以理解为对索引到的位置元素按照一定规则进行取值，取值所得再与`value`进行操作符运算，最终将运算结果进行赋值，所有索引增强赋值都不会改变原`Tensor`的`shape`。

> 当索引中有多个元素指向原张量的同一个位置时，该值的更新受底层算子限制，可能出现随机的情况。因此暂不支持索引中重复对张量中一个位置的值反复更新。详情请见:[TensorScatterUpdate 算子介绍](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.TensorScatterUpdate.html)
>
> 目前索引中包含 `True`、`False` 和 `None`的情况暂不支持.

- 规则与约束

    与索引赋值相比，增加了取值与运算的过程。取值过程中`index`的约束规则与索引取值中`index`相同，支持`int`、`bool`、`Tensor`、`Slice`、`Ellipsis`、`None`、`List`与`Tuple`。上述几种类型的数据中所包含`int`值，需在`[-dim_size, dim_size-1]`闭合区间内。
    运算过程中`value`的约束规则与索引赋值中`value`的约束规则相同，`value`类型需为(`Number`、`Tensor`、`List`、`Tuple`)之一，且`value`类型不是`Number`时， `value`的形状需要可以广播到`tensor_x[index]`的形状。

    示例如下：

    ```python
    tensor_x = Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
    tensor_y = Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
    tensor_x[[0, 1], 1:3] += 2
    tensor_y[[1], ...] -= [4, 3, 2, 1]
    ```

    结果如下：

    ```text
    tensor_x: Tensor(shape=[3, 4], dtype=Float32, value=[[0.0, 3.0, 4.0, 3.0], [4.0, 7.0, 8.0, 7.0], [8.0, 9.0, 10.0, 11.0]])
    tensor_y: Tensor(shape=[3, 4], dtype=Float32, value=[[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0], [8.0, 9.0, 10.0, 11.0]])
    ```

