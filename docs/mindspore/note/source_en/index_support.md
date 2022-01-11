# Tensor Index Support

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/note/source_en/index_support.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

Single-level and multi-level Tensor indexing is supported on both PyNative and Graph mode.

## Index values

The index value can be `int`, `bool`, `None`, `ellipsis`, `slice`, `Tensor`, `List`, or `Tuple`.

- `int` index value

    Single-level and multi-level `int` index values are supported. The single-level `int` index value is `tensor_x[int_index]`, and the multi-level `int` index value is `tensor_x[int_index0][int_index1]...`.

    The `int` index value is obtained on dimension 0 and is less than the length of dimension 0. After the position data corresponding to dimension 0 is obtained, dimension 0 is eliminated.

    For example, if a single-level `int` index value is obtained for a tensor whose `shape` is `(3, 4, 5)`, the obtained `shape` is `(4, 5)`.

    The multi-level index value can be understood as obtaining the current-level `int` index value based on the previous-level index value.

    For example:

    ```python
    from mindspore import Tensor
    import mindspore.numpy as np
    tensor_x = Tensor(np.arange(2 * 3 * 2).reshape((2, 3, 2)))
    data_single = tensor_x[0]
    data_multi = tensor_x[0][1]
    ```

    The result is as follows:

    ```text
    data_single: Tensor(shape=[3, 2], dtype=Int64, value=[[0, 1], [2, 3], [4, 5]])
    data_multi: Tensor(shape=[2], dtype=Int64, value=[2, 3])
    ```

- `bool` index value

    Single-level and multi-level `bool` index values are supported. The single-level `bool` index value is `tensor_x[True]`, and the multi-level `True` index value is `tensor_x[True][True]...`.

    The `True` index value operation is obtained on dimension 0. After all data is obtained, a dimension is extended on the `axis=0` axis. The length of the dimension is 1. `False` will introduce `0` in the shape, thus only `Ture` is supported now.

    For example, if a single-level `True` index value is obtained from a tensor whose `shape` is `(3, 4, 5)`, the obtained `shape` is `(1, 3, 4, 5)`.

    The multi-level index value can be understood as obtaining the current-level `bool` index value based on the previous-level index value.

    For example:

    ```python
    tensor_x = Tensor(np.arange(2 * 3 ).reshape((2, 3)))
    data_single = tensor_x[True]
    data_multi = tensor_x[True][True]
    ```

    The result is as follows:

    ```text
    data_single: Tensor(shape=[1, 2, 3], dtype=Int64, value=[[[0, 1, 2], [3, 4, 5]]])
    data_multi: Tensor(shape=[1, 0, 2, 3], dtype=Int64, value=[[[[0, 1, 2], [3  , 4, 5]]]])
    ```

- `None` index value

    The `None` index value is the same as the `True` index value. For details, see the `True` index value.

- `ellipsis` index value

    Single-level and multi-level `ellipsis` index values are supported. The single-level `ellipsis` index value is `tensor_x[...]`, and the multi-level `ellipsis` index value is `tensor_x[...][...]...`.

    The `ellipsis` index value is obtained on all dimensions to get the original data without any change. Generally, it is used as a component of the `Tuple` index. The `Tuple` index is described as follows.

    For example, if the `ellipsis` index value is obtained for a tensor whose `shape` is `(3, 4, 5)`, the obtained `shape` is still `(3, 4, 5)`.

    For example:

    ```python
    tensor_x = Tensor(np.arange(2 * 3 ).reshape((2, 3)))
    data_single = tensor_x[...]
    data_multi = tensor_x[...][...]
    ```

    The result is as follows:

    ```text
    data_single: Tensor(shape=[2, 3], dtype=Int64, value=[[0, 1, 2], [3, 4, 5]])
    data_multi: Tensor(shape=[2, 3], dtype=Int64, value=[[0, 1, 2], [3, 4, 5]])
    ```

- `slice` index value

    Single-level and multi-level `slice` index values are supported. The single-level `slice` index value is `tensor_x[slice_index]`, and the multi-level `slice` index value is `tensor_x[slice_index0][slice_index1]...`.

    The `slice` index value is obtained on dimension 0. The element of the sliced position on dimension 0 is obtained. The `slice` does not reduce the dimension even if the length is 1, which is different from the `int` index value.

    For example, `tensor_x[0:1:1] != tensor_x[0]`, because `shape_former = (1,) + shape_latter`.

    The multi-level index value can be understood as obtaining the current-level `slice` index value based on the previous-level index value.

    `slice` consists of `start`, `stop`, and `step`. The default value of `start` is 0, the default value of `stop` is the length of the dimension, and the default value of `step` is 1.

    Example: `tensor_x[:] == tensor_x[0:length:1]`.

    For example:

    ```python
    tensor_x = Tensor(np.arange(4 * 2 * 2).reshape((4, 2, 2)))
    data_single = tensor_x[1:4:2]
    data_multi = tensor_x[1:4:2][1:]
    ```

    The result is as follows:

    ```text
    data_single: Tensor(shape=[2, 2, 2], dtype=Int64, value=[[[4, 5], [6, 7]], [[12, 13], [14, 15]]])
    data_multi: Tensor(shape=[1, 2, 2], dtype=Int64, value=[[[12, 13], [14, 15]]])
    ```

- `Tensor` index value

    Single-level and multi-level `Tensor` index values are supported. The single-level `Tensor` index value is `tensor_x[tensor_index]`, and the multi-level `Tensor` index value is `tensor_x[tensor_index0][tensor_index1]...`.

    The `Tensor` index value is obtained on dimension 0, and the element in the corresponding position of dimension 0 is obtained.

    The data type of the `Tensor` index must be one of int8, int16, int32, and int64, the element cannot be a negative number, and the value must be less than the length of dimension 0.

    The `Tensor` index value is obtained by `data_shape = tensor_inde4x.shape + tensor_x.shape[1:]`.

    For example, if the index value is obtained for a tensor whose shape is `(6, 4, 5)` by using a tensor whose shape is `(2, 3)`, the obtained shape is `(2, 3, 4, 5)`.

    The multi-level index value can be understood as obtaining the current-level `Tensor` index value based on the previous-level index value.

    For example:

    ```python
    tensor_x = Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
    tensor_index0 = Tensor(np.array([[1, 2], [0, 3]]), mstype.int32)
    tensor_index1 = Tensor(np.array([[0, 0]]), mstype.int32)
    data_single = tensor_x[tensor_index0]
    data_multi = tensor_x[tensor_index0][tensor_index1]
    ```

    The result is as follows:

    ```text
    data_single: Tensor(shape=[2, 2, 2, 3], dtype=Int64, value=[[[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16 ,17]]], [[[0, 1, 2], [3, 4, 5]], [[18, 19, 20], [21, 22, 23]]]])
    data_multi: Tensor(shape=[1, 2, 2, 2, 3], dtype=Int64, value=[[[[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16 ,17]]], [[[0, 1, 2], [3, 4, 5]], [[18, 19, 20], [21, 22, 23]]]]]))
    ```

- `List` index value

    Single-level and multi-level `Tensor` index values are supported. The single-level `List` index value is `tensor_x[list_index]`, and the multi-level `List` index value is `tensor_x[list_index0][list_index1]...`.

    The `List` index value is obtained on dimension 0, and the element in the corresponding position of dimension 0 is obtained.

    The data type of the `List` index must be all bool, all int or mixed of them. The `List` elements of int type must be in the range of [`-dimension_shape`, `dimension_shape-1`] and the count of `List` elements with bool type must be the same as the `dimension_shape` of dimension 0 and will perform as to filter the corresponding element of the Tenson data. If the above two types appear simultaneously, the `List` elements with the bool type will be converted to `1/0` for `True/False`.

    The `Tensor` index value is obtained by `data_shape = tensor_inde4x.shape + tensor_x.shape[1:]`.

    For example, if the index value is obtained for a tensor whose shape is `(6, 4, 5)` by using a tensor whose shape is `(2, 3)`, the obtained shape is `(2, 3, 4, 5)`.

    The multi-level index value can be understood as obtaining the current-level `Tensor` index value based on the previous-level index value.

    For example:

    ```python
    tensor_x = Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
    tensor_index0 = Tensor(np.array([[1, 2], [0, 3]]), mstype.int32)
    tensor_index1 = Tensor(np.array([[0, 0]]), mstype.int32)
    data_single = tensor_x[tensor_index0]
    data_multi = tensor_x[tensor_index0][tensor_index1]
    ```

    The result is as follows:

    ```text
    data_single: Tensor(shape=[2, 2, 2, 3], dtype=Int64, value=[[[[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[0, 1], [2, 3]], [[12, 13], [14, 15]]]])
    data_multi: Tensor(shape=[1, 2, 2, 2, 3], dtype=Int64, value=[[[[[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[4, 5], [6, 7]], [[8, 9], [10, 11]]]]])
    ```

- `Tuple` index value

    The data type of the `Tuple` index can be `int`, `bool`, `None`, `slice`, `ellipsis`, `Tensor`, `List`, or `Tuple`. Single-level and multi-level `Tuple` index values are supported. For the single-level `Tuple` index, the value is `tensor_x[tuple_index]`. For the multi-level `Tuple` index, the value is `tensor_x[tuple_index0][tuple_index1]...`. The regulations of elements `List` and `Tuple` are the same as that of single index `List` index. The regulations of others are the same to the responding single element type.

    Elements in the `Tuple` index can be sort out by `Basic Index` or `Advanced Index`. `slice`, `ellipsis` and `None` are `Basic Index` and `int`, `bool`, `Tensor`, `List`, `Tuple` are `Advanced Index`. In the Getitem Progress, all the elements of the `Advanced Index` type will be broadcast to the same shape, and the final shape will be inserted to the first `Advanced Index` element's position if they are continuous, else they will be inserted to the `0` position.

    In the index, the `None` elements will expand the corresponding dimensions, `bool` elements will expand the corresponding dimension and be broadcast with the other `Advanced Index` element. The others elements except the type of `ellipsis`, `bool`, and `None`, will correspond to each position dimension. That is, the 0th element in `Tuple` operates the 0th dimension, and the 1st element operates the 1st dimension. The index rule of each element is the same as the index value rule of the element type.

    The `Tuple` index contains a maximum of one `ellipsis`. The first half of the `ellipsis` index elements correspond to the `Tensor` dimensions starting from the dimension 0, and the second half of the index elements correspond to the `Tensor` dimensions starting from the last dimension. If other dimensions are not specified, all dimensions are obtained.

    The data type of `Tensor` contained in the element must be one of (int8, int16, int32, int64). In addition, the value of `Tensor` element must be non-negative and less than the length of the operation dimension.

    For example, `tensor_x[0:3, 1, tensor_index] == tensor_x[(0:3, 1, tensor_index)]`, because `0:3, 1, tensor_index` is a `Tuple`.

    The multi-level index value can be understood as obtaining the current-level `Tuple` index value based on the previous-level index value.

    For example:

    ```python
    tensor_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)))
    tensor_index = Tensor(np.array([[1, 2, 1], [0, 3, 2]]), mstype.int32)
    data = tensor_x[1, 0:1, tensor_index]
    ```

    The result is as follows:

    ```text
    data: Tensor(shape=[2, 3, 1], dtype=Int64, value=[[[13], [14], [13]], [[12], [15], [14]]])
    ```

## Index value assignment

For a case like: `tensor_x[index] = value`, the type of the index can be `int`, `bool`, `ellipsis`, `slice`, `None`, `Tensor`, `List`, or`Tuple`.

The type of the assigned `value` can be `Number`, `Tuple`, `List`, or `Tensor`, the `value` will be converted to `Tensor` and casted to the same dtype as the original tensor (`tensor_x`) before being assigned.

When `value` is `Number`, all position elements obtained from the `tensor_x[index]` will be updated to `Number`.

When `value` is a tensor whose type is `Tuple`, `List` or `Tensor` and only contains `Number`, the `value.shape` needs to be able to be broadcasted to `tensor_x[index].shape`. After the `value`' is broadcasted and casted to `Tensor`, the elements with the position `tensor_x[index]` will be updated with the value `broadcast(Tensor(value))`.

When `value` is `Tuple/List`, and contains mixtures of `Number`, `Tuple`, `List` and `Tensor`, only one-dimensional `Tuple` and `List` are currently supported.

When `value` is `Tuple` or `List`, and contains `Tensor`, all the `non-Tensor` elements in `value` will be converted to `Tensor` first, and then these `Tensor` values are packed on the `axis=0` axis and become new `Tensor`. In this case, the value is assigned according to the rule of assigning the `value` to `Tensor`. All `Tensors` must have the same dtype.

Index value assignment can be understood as assigning values to indexed position elements based on certain rules. All index value assignment does not change the original `shape` of `Tensor`.

> If there are multiple index elements in indices that correspond to the same position, the value of that position in the output will be nondeterministic. For more details, please see:[TensorScatterUpdate](https://www.mindspore.cn/docs/api/en/r1.6/api_python/ops/mindspore.ops.TensorScatterUpdate.html)
>
> Only single-bracket indexing is supported (`tensor_x[index] = value`)， multi-bracket(`tensor_x[index1][index2]... = value`) is not supported.

- `int` index value assignment

    Single-level `int` index value assignments are supported. The single-level `int` index value assignment is `tensor_x[int_index] = u`.

    For example:

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_y = np.arange(2 *3).reshape((2, 3)).astype(np.float32)
    tensor_z = np.arange(2* 3).reshape((2, 3)).astype(np.float32)
    tensor_x[1] = 88.0
    tensor_y[1]= np.array([66, 88, 99]).astype(np.float32)
    tensor_z[1] = (66, np.array(88).astype(np.int64), 99)
    ```

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [66.0, 88.0, 99.0]])
    ```

- `bool` index value assignment

    Single-level `bool` index value assignments are supported. The single-level `int` index value assignment is `tensor_x[bool_index] = u`.

    For example:

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_z = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_x[True] = 88.0
    tensor_y[True]= np.array([66, 88, 99]).astype(np.float32)
    tensor_z[True] = (66, 88, 99)
    ```

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    ```

- `ellipsis` index value assignment

    Single-level `ellipsis` index value assignments are supported. The single-level `ellipsis` index value assignment is `tensor_x[...] = u`.

    For example:

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_z = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_x[...] = 88
    tensor_y[...]= np.array([[22, 44, 55], [22, 44, 55]])
    tensor_z[...] = ([11, 22, 33], [44, 55, 66])
    ```

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[22.0, 44.0, 55.0], [22.0, 44.0, 55.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]])
    ```

- `slice` index value assignment

    Single-level `slice` index value assignments are supported. The single-level `slice` index value assignment is `tensor_x[slice_index] = u`.

    For example:

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

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [11.0, 12.0, 13.0], [6.0, 7.0, 8.0]])
    tensor_k: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [6.0, 7.0, 8.0]])
    ```

- `None` index value assignment

    Single-level `None` index value assignments are supported. The single-level `int` index value assignment is `tensor_x[none_index] = u`.

    For example:

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_y = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_z = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    tensor_x[None] = 88.0
    tensor_y[None]= np.array([66, 88, 99]).astype(np.float32)
    tensor_z[None] = (66, 88, 99)
    ```

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[2, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    tensor_z: Tensor(shape=[2, 3], dtype=Float32, value=[[66.0, 88.0, 99.0], [66.0, 88.0, 99.0]])
    ```

- `Tensor` index value assignment

    Single-level `Tensor` index value assignments are supported. The single-level `Tensor` index value assignment is `tensor_x[tensor_index] = u`.

    Boolean `Tensor` index is not currently supported, only `mstype.int*` type is supported.

    For example:

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

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [3.0, 4.0, 5.0], [88.0, 88.0, 88.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [11.0, 12.0, 13.0]])
    ```

- `List` index value assignment

    single-level `List` index value assignments are supported. The single-level `List` index value assignment is `tensor_x[list_index] = u`.

    The `List` index value assignment is the same as that of the `List` index value.

    For example:

    ```python
    import mindspore.numpy as np
    tensor_x = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_y = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    tensor_index = np.array([[0, 1], [1, 0]]).astype(np.int32)
    tensor_x[[0,1]] = 88.0
    tensor_y[[True, False, False]] = np.array([11, 12, 13]).astype(np.float32)
    ```

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[88.0, 88.0, 88.0], [88.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[11.0, 12.0, 13.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    ```

- `Tuple` index value assignment

    single-level `Tuple` index value assignments are supported. The single-level `Tuple` index value assignment is `tensor_x[tuple_index] = u`.

    The `Tuple` index value assignment is the same as that of the `Tuple` index value, but `None` type is not supported now.

    For example:

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

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [3.0, 88.0, 88.0], [6.0, 7.0, 8.0]])
    tensor_y: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [88.0, 88.0, 5.0], [88.0, 88.0, 8.0]])
    tensor_z: Tensor(shape=[3, 3], dtype=Float32, value=[[0.0, 1.0, 2.0], [11.0, 12.0, 5.0], [11.0, 12.0, 8.0]])
    ```

## Index value augmented-assignment

Index value augmented-assignment supports seven augmented_assignment operations:  `+=`, `-=`, `*=`, `/=`, `%=`, `**=`, and `//=`. The rules and constraints of `index` and `value` are the same as index assignment. The index value supports eight types: `int`, `bool`, `ellipsis`, `slice`, `None`, `tensor`, `list` and `tuple`. The assignment value supports four types: `Number`, `Tensor`, `Tuple` and `List`.

Index value augmented-assignment can be regarded as taking the value of the position elements to be indexed according to certain rules, and then performing operator operation with `value`. Finally, assign the operation result to the origin `Tensor`. All index augmented-assignments will not change the `shape` of the original `Tensor`.

> If there are multiple index elements in indices that correspond to the same position, the value of that position in the output will be nondeterministic. For more details, please see:[TensorScatterUpdate](https://www.mindspore.cn/docs/api/en/r1.6/api_python/ops/mindspore.ops.TensorScatterUpdate.html)
>
> Currently indices that contain `True`, `False` and `None` are not supported.

- Rules and constraints:

    Compared with index assignment, the process of value and operation is increased. The constraint rules of `index` are the same as `index` in Index Value, and support `Int`, `Bool`, `Tensor`, `Slice`, `Ellipse`, `None`, `List` and `Tuple`. The values of `Int` contained in the above types of data should be in `[-dim_size, dim_size-1]` within the closed range.

    The constraint rules of `value` in the operation process are the same as those of `value` in index assignment. The type of `value` needs to be one of (`Number`, `Tensor`, `List`, `Tuple`). And if `value`'s type is not `number`, `value.shape` should be able to broadcast to `tensor_x[index].shape`.

    For example:

    ```python
    tensor_x = Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
    tensor_y = Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
    tensor_x[[0, 1], 1:3] += 2
    tensor_y[[1], ...] -= [4, 3, 2, 1]
    ```

    The result is as follows:

    ```text
    tensor_x: Tensor(shape=[3, 4], dtype=Float32, value=[[0.0, 3.0, 4.0, 3.0], [4.0, 7.0, 8.0, 7.0], [8.0, 9.0, 10.0, 11.0]])
    tensor_y: Tensor(shape=[3, 4], dtype=Float32, value=[[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0], [8.0, 9.0, 10.0, 11.0]])
    ```

