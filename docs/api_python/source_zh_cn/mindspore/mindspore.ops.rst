mindspore.ops
=============

.. automodule:: mindspore.ops

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, please refer to the link `<https://gitee.com/mindspore/docs/blob/master/resource/api_updates/ops_api_updates.md>`_.

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
     - :class:`mindspore.ops.Primitive` ('array_reduce')
   * - mindspore.ops.array_to_scalar
     - :class:`mindspore.ops.Primitive` ('array_to_scalar')
   * - mindspore.ops.assign
     - Refer to :class:`mindspore.ops.Assign`.
   * - mindspore.ops.assign_add
     - Refer to :class:`mindspore.ops.AssignAdd`.
   * - mindspore.ops.assign_sub
     - Refer to :class:`mindspore.ops.AssignSub`.
   * - mindspore.ops.bool_and
     - :class:`mindspore.ops.Primitive` ('bool_and')
   * - mindspore.ops.bool_eq
     - :class:`mindspore.ops.Primitive` ('bool_eq')
   * - mindspore.ops.bool_not
     - :class:`mindspore.ops.Primitive` ('bool_not')
   * - mindspore.ops.bool_or
     - :class:`mindspore.ops.Primitive` ('bool_or')
   * - mindspore.ops.cast
     - Refer to :class:`mindspore.ops.Cast`.
   * - mindspore.ops.distribute
     - :class:`mindspore.ops.Primitive` ('distribute')
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
     - :class:`mindspore.ops.Primitive` ('hastype')
   * - mindspore.ops.in_dict
     - :class:`mindspore.ops.Primitive` ('in_dict')
   * - mindspore.ops.is_not
     - :class:`mindspore.ops.Primitive` ('is_not')
   * - mindspore.ops.is\_
     - :class:`mindspore.ops.Primitive` ('is\_')
   * - mindspore.ops.isconstant
     - :class:`mindspore.ops.Primitive` ('is_constant')
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
     - :class:`mindspore.ops.Primitive` ('mixed_precision_cast')
   * - mindspore.ops.neg_tensor
     - Refer to :class:`mindspore.ops.Neg`.
   * - mindspore.ops.not_equal
     - Refer to :class:`mindspore.ops.NotEqual`.
   * - mindspore.ops.not_in_dict
     - :class:`mindspore.ops.Primitive` ('not_in_dict')
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
     - :class:`mindspore.ops.Primitive` ('scalar_add')
   * - mindspore.ops.scalar_cast
     - Refer to :class:`mindspore.ops.ScalarCast`.
   * - mindspore.ops.scalar_div
     - :class:`mindspore.ops.Primitive` ('scalar_div')
   * - mindspore.ops.scalar_eq
     - :class:`mindspore.ops.Primitive` ('scalar_eq')
   * - mindspore.ops.scalar_floordiv
     - :class:`mindspore.ops.Primitive` ('scalar_floordiv')
   * - mindspore.ops.scalar_ge
     - :class:`mindspore.ops.Primitive` ('scalar_ge')
   * - mindspore.ops.scalar_gt
     - :class:`mindspore.ops.Primitive` ('scalar_gt')
   * - mindspore.ops.scalar_le
     - :class:`mindspore.ops.Primitive` ('scalar_le')
   * - mindspore.ops.scalar_log
     - :class:`mindspore.ops.Primitive` ('scalar_log')
   * - mindspore.ops.scalar_lt
     - :class:`mindspore.ops.Primitive` ('scalar_lt')
   * - mindspore.ops.scalar_mod
     - :class:`mindspore.ops.Primitive` ('scalar_mod')
   * - mindspore.ops.scalar_mul
     - :class:`mindspore.ops.Primitive` ('scalar_mul')
   * - mindspore.ops.scalar_ne
     - :class:`mindspore.ops.Primitive` ('scalar_ne')
   * - mindspore.ops.scalar_pow
     - :class:`mindspore.ops.Primitive` ('scalar_pow')
   * - mindspore.ops.scalar_sub
     - :class:`mindspore.ops.Primitive` ('scalar_sub')
   * - mindspore.ops.scalar_to_array
     - Refer to :class:`mindspore.ops.ScalarToArray`.
   * - mindspore.ops.scalar_to_tensor
     - Refer to :class:`mindspore.ops.ScalarToTensor`.
   * - mindspore.ops.scalar_uadd
     - :class:`mindspore.ops.Primitive` ('scalar_uadd')
   * - mindspore.ops.scalar_usub
     - :class:`mindspore.ops.Primitive` ('scalar_usub')
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
     - Disable update during back propagation.
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

    mindspore.ops.AiCPURegOp
    mindspore.ops.DataType
    mindspore.ops.op_info_register
    mindspore.ops.TBERegOp
