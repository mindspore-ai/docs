mindspore
=========

.. class:: mindspore.dtype
  
  Create a data type object of MindSpore.
  
  The actual path of ``dtype`` is ``/mindspore/common/dtype.py``.
  Run the following command to import the package:
  
  .. code-block::
  
      from mindspore import dtype as mstype
  
  * **Numeric Type**
  
    Currently, MindSpore supports ``Int`` type, ``Uint`` type and ``Float`` type.
    The following table lists the details.
  
    ==============================================   =============================
    Definition                                        Description
    ==============================================   =============================
    ``mindspore.int8`` ,  ``mindspore.byte``         8-bit integer
    ``mindspore.int16`` ,  ``mindspore.short``       16-bit integer 
    ``mindspore.int32`` ,  ``mindspore.intc``        32-bit integer
    ``mindspore.int64`` ,  ``mindspore.intp``        64-bit integer
    ``mindspore.uint8`` ,  ``mindspore.ubyte``       unsigned 8-bit integer
    ``mindspore.uint16`` ,  ``mindspore.ushort``     unsigned 16-bit integer
    ``mindspore.uint32`` ,  ``mindspore.uintc``      unsigned 32-bit integer
    ``mindspore.uint64`` ,  ``mindspore.uintp``      unsigned 64-bit integer
    ``mindspore.float16`` ,  ``mindspore.half``      16-bit floating-point number
    ``mindspore.float32`` ,  ``mindspore.single``    32-bit floating-point number
    ``mindspore.float64`` ,  ``mindspore.double``    64-bit floating-point number
    ==============================================   =============================
  
  * **Other Type**
  
    For other defined types, see the following table.
  
    ============================   =================
    Type                            Description
    ============================   =================
    ``tensor``                      MindSpore's ``tensor`` type. Data format uses NCHW. For details, see `tensor <https://www.gitee.com/mindspore/mindspore/blob/r1.1/mindspore/common/tensor.py>`_.
    ``MetaTensor``                  A tensor only has data type and shape. For details, see `MetaTensor <https://www.gitee.com/mindspore/mindspore/blob/r1.1/mindspore/common/parameter.py>`_.
    ``bool_``                       Boolean ``True`` or ``False``.
    ``int_``                        Integer scalar.
    ``uint``                        Unsigned integer scalar.
    ``float_``                      Floating-point scalar.
    ``number``                      Number, including ``int_`` , ``uint`` , ``float_`` and ``bool_`` .
    ``list_``                       List constructed by ``tensor`` , such as ``List[T0,T1,...,Tn]`` , where the element ``Ti`` can be of different types.
    ``tuple_``                      Tuple constructed by ``tensor`` , such as ``Tuple[T0,T1,...,Tn]`` , where the element ``Ti`` can be of different types.
    ``function``                    Function. Return in two ways, when function is not None, returns Func directly, the other returns Func(args: List[T0,T1,...,Tn], retval: T) when function is None.
    ``type_type``                   Type definition of type.
    ``type_none``                   No matching return type, corresponding to the ``type(None)`` in Python.
    ``symbolic_key``                The value of a variable is used as a key of the variable in ``env_type`` .
    ``env_type``                    Used to store the gradient of the free variable of a function, where the key is the ``symbolic_key`` of the free variable's node and the value is the gradient.
    ============================   =================
  
  * **Tree Topology**
  
    The relationships of the above types are as follows:
  
    .. code-block::
    
    
        └─────── number
            │   ├─── bool_
            │   ├─── int_
            │   │   ├─── int8, byte
            │   │   ├─── int16, short
            │   │   ├─── int32, intc
            │   │   └─── int64, intp
            │   ├─── uint
            │   │   ├─── uint8, ubyte
            │   │   ├─── uint16, ushort
            │   │   ├─── uint32, uintc
            │   │   └─── uint64, uintp
            │   └─── float_
            │       ├─── float16
            │       ├─── float32
            │       └─── float64
            ├─── tensor
            │   ├─── Array[Float32]
            │   └─── ...
            ├─── list_
            │   ├─── List[Int32,Float32]
            │   └─── ...
            ├─── tuple_
            │   ├─── Tuple[Int32,Float32]
            │   └─── ...
            ├─── function
            │   ├─── Func
            │   ├─── Func[(Int32, Float32), Int32]
            │   └─── ...
            ├─── MetaTensor
            ├─── type_type
            ├─── type_none
            ├─── symbolic_key
            └─── env_type
  
.. automodule:: mindspore
    :members:
