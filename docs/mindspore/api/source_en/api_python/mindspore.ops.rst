mindspore.ops
=============

.. automodule:: mindspore.ops

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, please refer to the link `<https://gitee.com/mindspore/docs/blob/r1.3/resource/api_updates/ops_api_updates.md>`_.

.. include:: operations.rst

composite
---------

The composite operators are the pre-defined combination of operators.

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.batch_dot
    mindspore.ops.clip_by_global_norm
    mindspore.ops.clip_by_value
    mindspore.ops.core
    mindspore.ops.count_nonzero
    mindspore.ops.dot
    mindspore.ops.gamma
    mindspore.ops.GradOperation
    mindspore.ops.HyperMap
    mindspore.ops.laplace
    mindspore.ops.matmul
    mindspore.ops.multinomial
    mindspore.ops.MultitypeFuncGraph
    mindspore.ops.normal
    mindspore.ops.poisson
    mindspore.ops.repeat_elements
    mindspore.ops.sequence_mask
    mindspore.ops.tensor_dot
    mindspore.ops.uniform

functional
----------

The functional operators are the pre-instantiated Primitive operators, which can be used directly as a function.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.add
     - Refer to :class:`mindspore.ops.Add`.
   * - mindspore.ops.addn
     - Refer to :class:`mindspore.ops.AddN`.
   * - mindspore.ops.array_reduce
     - Reduce the dimension of the array.
   * - mindspore.ops.array_to_scalar
     - Convert the array to a scalar.
   * - mindspore.ops.assign
     - Refer to :class:`mindspore.ops.Assign`.
   * - mindspore.ops.assign_add
     - Refer to :class:`mindspore.ops.AssignAdd`.
   * - mindspore.ops.assign_sub
     - Refer to :class:`mindspore.ops.AssignSub`.
   * - mindspore.ops.bool_and
     - Calculate the result of logical AND operation.
   * - mindspore.ops.bool_eq
     - Determine whether the Boolean values are equal.
   * - mindspore.ops.bool_not
     - Calculate the result of logical NOT operation.
   * - mindspore.ops.bool_or
     - Calculate the result of logical OR operation.
   * - mindspore.ops.cast
     - Refer to :class:`mindspore.ops.Cast`.
   * - mindspore.ops.dtype
     - Refer to :class:`mindspore.ops.DType`.
   * - mindspore.ops.equal
     - Refer to :class:`mindspore.ops.Equal`.
   * - mindspore.ops.expand_dims
     - Refer to :class:`mindspore.ops.ExpandDims`.
   * - mindspore.ops.fill
     - Refer to :class:`mindspore.ops.Fill`.
   * - mindspore.ops.gather
     - Refer to :class:`mindspore.ops.Gather`.
   * - mindspore.ops.gather_nd
     - Refer to :class:`mindspore.ops.GatherNd`.
   * - mindspore.ops.hastype
     - Determine whether the object has the specified type.
   * - mindspore.ops.in_dict
     - Determine whether the object is in the dict.
   * - mindspore.ops.is_not
     - Determine whether the input is not the same as the other one.
   * - mindspore.ops.is\_
     - Determine whether the input is the same as the other one.
   * - mindspore.ops.isconstant
     - Determine whether the object is constant.
   * - mindspore.ops.isinstance\_
     - Refer to :class:`mindspore.ops.IsInstance`.
   * - mindspore.ops.issubclass\_
     - Refer to :class:`mindspore.ops.IsSubClass`.
   * - mindspore.ops.logical_and
     - Refer to :class:`mindspore.ops.LogicalAnd`.
   * - mindspore.ops.logical_not
     - Refer to :class:`mindspore.ops.LogicalNot`.
   * - mindspore.ops.logical_or
     - Refer to :class:`mindspore.ops.LogicalOr`.
   * - mindspore.ops.make_row_tensor
     - Generate row tensor.
   * - mindspore.ops.make_sparse_tensor
     - Generate sparse tensor.
   * - mindspore.ops.mixed_precision_cast
     - A temporary ops for mixed precision will be converted to cast after the step of compiling.
   * - mindspore.ops.neg_tensor
     - Refer to :class:`mindspore.ops.Neg`.
   * - mindspore.ops.not_equal
     - Refer to :class:`mindspore.ops.NotEqual`.
   * - mindspore.ops.not_in_dict
     - Determine whether the object is not in the dict.
   * - mindspore.ops.ones_like
     - Refer to :class:`mindspore.ops.OnesLike`.
   * - mindspore.ops.print\_
     - Refer to :class:`mindspore.ops.Print`.
   * - mindspore.ops.rank
     - Refer to :class:`mindspore.ops.Rank`.
   * - mindspore.ops.reduced_shape
     - Calculate the shape of the reduction operator.
   * - mindspore.ops.reshape
     - Refer to :class:`mindspore.ops.Reshape`.
   * - mindspore.ops.row_tensor_get_dense_shape
     - Get corresponding dense shape of row tensor.
   * - mindspore.ops.row_tensor_get_indices
     - Get indices of row tensor.
   * - mindspore.ops.row_tensor_get_values
     - Get values of row tensor.
   * - mindspore.ops.same_type_shape
     - Refer to :class:`mindspore.ops.SameTypeShape`.
   * - mindspore.ops.scalar_add
     - Get the sum of two numbers.
   * - mindspore.ops.scalar_cast
     - Refer to :class:`mindspore.ops.ScalarCast`.
   * - mindspore.ops.scalar_div
     - Get the quotient of dividing the first input number by the second input number.
   * - mindspore.ops.scalar_eq
     - Determine whether two numbers are equal.
   * - mindspore.ops.scalar_floordiv
     - Divide the first input number by the second input number and round down to the closest integer.
   * - mindspore.ops.scalar_ge
     - Determine whether the number is greater than or equal to another number.
   * - mindspore.ops.scalar_gt
     - Determine whether the number is greater than another number.
   * - mindspore.ops.scalar_le
     - Determine whether the number is less than or equal to another number.
   * - mindspore.ops.scalar_log
     - Get the natural logarithm of the input number.
   * - mindspore.ops.scalar_lt
     - Determine whether the number is less than another number.
   * - mindspore.ops.scalar_mod
     - Get the remainder of dividing the first input number by the second input number.
   * - mindspore.ops.scalar_mul
     - Get the product of the input two numbers.
   * - mindspore.ops.scalar_ne
     - Determine whether two numbers are not equal.
   * - mindspore.ops.scalar_pow
     - Compute a number to the power of the second input number.
   * - mindspore.ops.scalar_sub
     - Subtract the second input number from the first input number.
   * - mindspore.ops.scalar_to_array
     - Refer to :class:`mindspore.ops.ScalarToArray`.
   * - mindspore.ops.scalar_to_tensor
     - Refer to :class:`mindspore.ops.ScalarToTensor`.
   * - mindspore.ops.scalar_uadd
     - Get the positive value of the input number.
   * - mindspore.ops.scalar_usub
     - Get the negative value of the input number.
   * - mindspore.ops.scatter_nd
     - Refer to :class:`mindspore.ops.ScatterNd`.
   * - mindspore.ops.scatter_nd_update
     - Refer to :class:`mindspore.ops.ScatterNdUpdate`.
   * - mindspore.ops.scatter_update
     - Refer to :class:`mindspore.ops.ScatterUpdate`.
   * - mindspore.ops.select
     - Refer to :class:`mindspore.ops.Select`.
   * - mindspore.ops.shape
     - Refer to :class:`mindspore.ops.Shape`.
   * - mindspore.ops.shape_mul
     - The input of shape_mul must be shape multiply elements in tuple(shape).
   * - mindspore.ops.size
     - Refer to :class:`mindspore.ops.Size`.
   * - mindspore.ops.sparse_tensor_get_dense_shape
     - Get corresponding dense shape of sparse tensor.
   * - mindspore.ops.sparse_tensor_get_indices
     - Get indices of sparse tensor.
   * - mindspore.ops.sparse_tensor_get_values
     - Get values of sparse tensor.
   * - mindspore.ops.sqrt
     - Refer to :class:`mindspore.ops.Sqrt`.
   * - mindspore.ops.square
     - Refer to :class:`mindspore.ops.Square`.
   * - mindspore.ops.stack
     - Refer to :class:`mindspore.ops.Stack`.
   * - mindspore.ops.stop_gradient
     - Disable update during back propagation. (`stop_gradient <https://www.mindspore.cn/tutorials/en/r1.3/autograd.html#stop-gradient>`_)
   * - mindspore.ops.strided_slice
     - Refer to :class:`mindspore.ops.StridedSlice`.
   * - mindspore.ops.string_concat
     - Concatenate two strings.
   * - mindspore.ops.string_eq
     - Determine if two strings are equal.
   * - mindspore.ops.tensor_div
     - Refer to :class:`mindspore.ops.RealDiv`.
   * - mindspore.ops.tensor_floordiv
     - Refer to :class:`mindspore.ops.FloorDiv`.
   * - mindspore.ops.tensor_ge
     - Refer to :class:`mindspore.ops.GreaterEqual`.
   * - mindspore.ops.tensor_gt
     - Refer to :class:`mindspore.ops.Greater`.
   * - mindspore.ops.tensor_le
     - Refer to :class:`mindspore.ops.LessEqual`.
   * - mindspore.ops.tensor_lt
     - Refer to :class:`mindspore.ops.Less`.
   * - mindspore.ops.tensor_mod
     - Refer to :class:`mindspore.ops.FloorMod`.
   * - mindspore.ops.tensor_mul
     - Refer to :class:`mindspore.ops.Mul`.
   * - mindspore.ops.tensor_pow
     - Refer to :class:`mindspore.ops.Pow`.
   * - mindspore.ops.tensor_sub
     - Refer to :class:`mindspore.ops.Sub`.
   * - mindspore.ops.tile
     - Refer to :class:`mindspore.ops.Tile`.
   * - mindspore.ops.tuple_to_array
     - Refer to :class:`mindspore.ops.TupleToArray`.
   * - mindspore.ops.typeof
     - Get type of object.
   * - mindspore.ops.zeros_like
     - Refer to :class:`mindspore.ops.ZerosLike`.

primitive
---------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.constexpr
    mindspore.ops.prim_attr_register
    mindspore.ops.Primitive
    mindspore.ops.PrimitiveWithCheck
    mindspore.ops.PrimitiveWithInfer

vm_impl_registry
----------------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.get_vm_impl_fn

op_info_register
----------------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.DataType
    mindspore.ops.op_info_register
    mindspore.ops.TBERegOp