mindspore.ops
=============

.. automodule:: mindspore.ops

.. include:: operations.rst

composite
---------

The composite operators are the pre-defined combination of operators.

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.add_flags
    mindspore.ops.clip_by_global_norm
    mindspore.ops.clip_by_value
    mindspore.ops.core
    mindspore.ops.count_nonzero
    mindspore.ops.gamma
    mindspore.ops.GradOperation
    mindspore.ops.HyperMap
    mindspore.ops.laplace
    mindspore.ops.multinomial
    mindspore.ops.MultitypeFuncGraph
    mindspore.ops.normal
    mindspore.ops.poisson
    mindspore.ops.TensorDot
    mindspore.ops.uniform

functional
----------

The functional operators are the pre-instantiated Primitive operators, which can be used directly as a function.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - operations
   * - mindspore.ops.addn
     - :class:`mindspore.ops.AddN`
   * - mindspore.ops.array_reduce
     - :class:`mindspore.ops.Primitive` ('array_reduce')               
   * - mindspore.ops.array_to_scalar
     - :class:`mindspore.ops.Primitive` ('array_to_scalar')               
   * - mindspore.ops.assign
     - :class:`mindspore.ops.Assign`
   * - mindspore.ops.assign_add
     - :class:`mindspore.ops.AssignAdd`
   * - mindspore.ops.assign_sub
     - :class:`mindspore.ops.AssignSub`
   * - mindspore.ops.bool_and
     - :class:`mindspore.ops.Primitive` ('bool_and')               
   * - mindspore.ops.bool_eq
     - :class:`mindspore.ops.Primitive` ('bool_eq')               
   * - mindspore.ops.bool_not
     - :class:`mindspore.ops.Primitive` ('bool_not')               
   * - mindspore.ops.bool_or
     - :class:`mindspore.ops.Primitive` ('bool_or')               
   * - mindspore.ops.cast
     - :class:`mindspore.ops.Cast`
   * - mindspore.ops.control_depend
     - :class:`mindspore.ops.ControlDepend`
   * - mindspore.ops.distribute
     - :class:`mindspore.ops.Primitive` ('distribute')               
   * - mindspore.ops.dtype
     - :class:`mindspore.ops.DType`
   * - mindspore.ops.equal
     - :class:`mindspore.ops.Equal`
   * - mindspore.ops.expand_dims
     - :class:`mindspore.ops.ExpandDims`
   * - mindspore.ops.fill
     - :class:`mindspore.ops.Fill`
   * - mindspore.ops.gather
     - :class:`mindspore.ops.GatherV2`
   * - mindspore.ops.gather_nd
     - :class:`mindspore.ops.GatherNd`
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
     - :class:`mindspore.ops.IsInstance`
   * - mindspore.ops.issubclass\_
     - :class:`mindspore.ops.IsSubClass`
   * - mindspore.ops.logical_and
     - :class:`mindspore.ops.LogicalAnd`
   * - mindspore.ops.logical_not
     - :class:`mindspore.ops.LogicalNot`
   * - mindspore.ops.logical_or
     - :class:`mindspore.ops.LogicalOr`
   * - mindspore.ops.make_row_tensor
     - :class:`mindspore.ops.Primitive` ('MakeRowTensor')               
   * - mindspore.ops.make_sparse_tensor
     - :class:`mindspore.ops.Primitive` ('MakeSparseTensor')               
   * - mindspore.ops.mixed_precision_cast
     - :class:`mindspore.ops.Primitive` ('mixed_precision_cast')               
   * - mindspore.ops.neg_tensor
     - :class:`mindspore.ops.Neg`
   * - mindspore.ops.not_equal
     - :class:`mindspore.ops.NotEqual`
   * - mindspore.ops.not_in_dict
     - :class:`mindspore.ops.Primitive` ('not_in_dict')               
   * - mindspore.ops.ones
     - :class:`mindspore.ops.Ones`
   * - mindspore.ops.ones_like
     - :class:`mindspore.ops.OnesLike`
   * - mindspore.ops.pack
     - :class:`mindspore.ops.Pack`
   * - mindspore.ops.print
     - :class:`mindspore.ops.Print`
   * - mindspore.ops.print\_
     - :class:`mindspore.ops.Print`
   * - mindspore.ops.rank
     - :class:`mindspore.ops.Rank`
   * - mindspore.ops.reduced_shape
     - :class:`mindspore.ops.Primitive` ('reduced_shape')               
   * - mindspore.ops.reshape
     - :class:`mindspore.ops.Reshape`
   * - mindspore.ops.row_tensor_get_dense_shape
     - :class:`mindspore.ops.Primitive` ('RowTensorGetDenseShape')               
   * - mindspore.ops.row_tensor_get_indices
     - :class:`mindspore.ops.Primitive` ('RowTensorGetIndices')               
   * - mindspore.ops.row_tensor_get_values
     - :class:`mindspore.ops.Primitive` ('RowTensorGetValues')               
   * - mindspore.ops.same_type_shape
     - :class:`mindspore.ops.SameTypeShape`
   * - mindspore.ops.scalar_add
     - :class:`mindspore.ops.Primitive` ('scalar_add')               
   * - mindspore.ops.scalar_cast
     - :class:`mindspore.ops.ScalarCast`
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
     - :class:`mindspore.ops.ScalarToArray`
   * - mindspore.ops.scalar_to_tensor
     - :class:`mindspore.ops.ScalarToTensor`
   * - mindspore.ops.scalar_uadd
     - :class:`mindspore.ops.Primitive` ('scalar_uadd')               
   * - mindspore.ops.scalar_usub
     - :class:`mindspore.ops.Primitive` ('scalar_usub')               
   * - mindspore.ops.scatter_nd
     - :class:`mindspore.ops.ScatterNd`
   * - mindspore.ops.scatter_nd_update
     - :class:`mindspore.ops.ScatterNdUpdate`
   * - mindspore.ops.scatter_update
     - :class:`mindspore.ops.ScatterUpdate`
   * - mindspore.ops.select
     - :class:`mindspore.ops.Select`
   * - mindspore.ops.shape
     - :class:`mindspore.ops.Shape`
   * - mindspore.ops.shape_mul
     - :class:`mindspore.ops.Primitive` ('shape_mul')               
   * - mindspore.ops.size
     - :class:`mindspore.ops.Size`
   * - mindspore.ops.sparse_tensor_get_dense_shape
     - :class:`mindspore.ops.Primitive` ('SparseTensorGetDenseShape')               
   * - mindspore.ops.sparse_tensor_get_indices
     - :class:`mindspore.ops.Primitive` ('SparseTensorGetIndices')               
   * - mindspore.ops.sparse_tensor_get_values
     - :class:`mindspore.ops.Primitive` ('SparseTensorGetValues')               
   * - mindspore.ops.sqrt
     - :class:`mindspore.ops.Sqrt`
   * - mindspore.ops.square
     - :class:`mindspore.ops.Square`
   * - mindspore.ops.stop_gradient
     - :class:`mindspore.ops.Primitive` ('stop_gradient')               
   * - mindspore.ops.strided_slice
     - :class:`mindspore.ops.StridedSlice`
   * - mindspore.ops.string_concat
     - :class:`mindspore.ops.Primitive` ('string_concat')               
   * - mindspore.ops.string_eq
     - :class:`mindspore.ops.Primitive` ('string_equal')               
   * - mindspore.ops.tensor_add
     - :class:`mindspore.ops.TensorAdd`
   * - mindspore.ops.tensor_div
     - :class:`mindspore.ops.RealDiv`
   * - mindspore.ops.tensor_floordiv
     - :class:`mindspore.ops.FloorDiv`
   * - mindspore.ops.tensor_ge
     - :class:`mindspore.ops.GreaterEqual`
   * - mindspore.ops.tensor_gt
     - :class:`mindspore.ops.Greater`
   * - mindspore.ops.tensor_le
     - :class:`mindspore.ops.LessEqual`
   * - mindspore.ops.tensor_lt
     - :class:`mindspore.ops.Less`
   * - mindspore.ops.tensor_mod
     - :class:`mindspore.ops.FloorMod`
   * - mindspore.ops.tensor_mul
     - :class:`mindspore.ops.Mul`
   * - mindspore.ops.tensor_pow
     - :class:`mindspore.ops.Pow`
   * - mindspore.ops.tensor_sub
     - :class:`mindspore.ops.Sub`
   * - mindspore.ops.tile
     - :class:`mindspore.ops.Tile`
   * - mindspore.ops.tuple_to_array
     - :class:`mindspore.ops.TupleToArray`
   * - mindspore.ops.typeof
     - :class:`mindspore.ops.Primitive` ('typeof')               
   * - mindspore.ops.zeros
     - :class:`mindspore.ops.Zeros`
   * - mindspore.ops.zeros_like
     - :class:`mindspore.ops.ZerosLike`

primitive
---------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.constexpr
    mindspore.ops.prim_attr_register
    mindspore.ops.Primitive
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
