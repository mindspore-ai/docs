mindspore.ops
=============

.. automodule:: mindspore.ops

.. include:: operations.rst

composite
---------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.add_flags
    mindspore.ops.clip_by_value
    mindspore.ops.core
    mindspore.ops.gamma
    mindspore.ops.GradOperation
    mindspore.ops.HyperMap
    mindspore.ops.laplace
    mindspore.ops.multinomial
    mindspore.ops.MultitypeFuncGraph
    mindspore.ops.normal
    mindspore.ops.poisson
    mindspore.ops.uniform

functional
----------

.. list-table:: 
   :widths: 50 50
   :header-rows: 1

   * - functional
     - operations
   * - mindspore.ops.addn
     - :class:`mindspore.ops.AddN`
   * - mindspore.ops.assign
     - :class:`mindspore.ops.Assign`
   * - mindspore.ops.assign_sub
     - :class:`mindspore.ops.AssignSub`
   * - mindspore.ops.cast
     - :class:`mindspore.ops.Cast`
   * - mindspore.ops.control_depend
     - :class:`mindspore.ops.ControlDepend`
   * - mindspore.ops.dtype
     - :class:`mindspore.ops.DType`
   * - mindspore.ops.equal
     - :class:`mindspore.ops.Equal`
   * - mindspore.ops.expand_dims
     - :class:`mindspore.ops.ExpandDims`
   * - mindspore.ops.fill
     - :class:`mindspore.ops.Fill`
   * - mindspore.ops.gather_nd
     - :class:`mindspore.ops.GatherNd`
   * - mindspore.ops.gather
     - :class:`mindspore.ops.GatherV2`
   * - mindspore.ops.logical_and
     - :class:`mindspore.ops.LogicalAnd`
   * - mindspore.ops.logical_not
     - :class:`mindspore.ops.LogicalNot`
   * - mindspore.ops.logical_or
     - :class:`mindspore.ops.LogicalOr`
   * - mindspore.ops.not_equal
     - :class:`mindspore.ops.NotEqual`
   * - mindspore.ops.ones_like
     - :class:`mindspore.ops.OnesLike`
   * - mindspore.ops.pack
     - :class:`mindspore.ops.Pack`
   * - mindspore.ops.tensor_pow
     - :class:`mindspore.ops.Pow`
   * - mindspore.ops.print
     - :class:`mindspore.ops.Print`
   * - mindspore.ops.rank
     - :class:`mindspore.ops.Rank`
   * - mindspore.ops.reshape
     - :class:`mindspore.ops.Reshape`
   * - mindspore.ops.scatter_nd
     - :class:`mindspore.ops.ScatterNd`
   * - mindspore.ops.select
     - :class:`mindspore.ops.Select`
   * - mindspore.ops.shape
     - :class:`mindspore.ops.Shape`
   * - mindspore.ops.size
     - :class:`mindspore.ops.Size`
   * - mindspore.ops.sqrt
     - :class:`mindspore.ops.Sqrt`
   * - mindspore.ops.square
     - :class:`mindspore.ops.Square`
   * - mindspore.ops.tensor_add
     - :class:`mindspore.ops.TensorAdd`
   * - mindspore.ops.tile
     - :class:`mindspore.ops.Tile`

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
