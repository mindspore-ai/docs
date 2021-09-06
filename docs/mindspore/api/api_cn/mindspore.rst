mindspore
=========

.. py:class:: mindspore.dtype
  
  创建一个MindSpore数据类型的对象。

  ``dtype``的实际路径为``/mindspore/common/dtype.py``，运行以下命令导入环境：
  
  .. code-block::
  
      from mindspore import dtype as mstype
  
  * **数值型**

    目前，MindSpore支持 ``Int``，``Uint`` 和 ``Float`` 数据类型，详情请参照以下表格。

    ==============================================   =============================
    定义                                              描述
    ==============================================   =============================
    ``mindspore.int8`` ,  ``mindspore.byte``         8位整型数
    ``mindspore.int16`` ,  ``mindspore.short``       16位整型数 
    ``mindspore.int32`` ,  ``mindspore.intc``        32位整型数
    ``mindspore.int64`` ,  ``mindspore.intp``        64位整型数
    ``mindspore.uint8`` ,  ``mindspore.ubyte``       无符号8位整型数
    ``mindspore.uint16`` ,  ``mindspore.ushort``     无符号16位整型数
    ``mindspore.uint32`` ,  ``mindspore.uintc``      无符号32位整型数
    ``mindspore.uint64`` ,  ``mindspore.uintp``      无符号64位整型数
    ``mindspore.float16`` ,  ``mindspore.half``      16位浮点数
    ``mindspore.float32`` ,  ``mindspore.single``    32位浮点数
    ``mindspore.float64`` ,  ``mindspore.double``    64位浮点数
    ``mindspore.complex64``                          64位复数
    ``mindspore.complex128``                         128位复数
    ==============================================   =============================
  
  * **其他类型**
  
    除数值型以外的其他数据类型，请参照以下表格。
  
    ============================   =================
    类型                            描述
    ============================   =================
    ``tensor``                      MindSpore中的张量类型。数据格式采用NCHW。详情请参考 `tensor <https://www.gitee.com/mindspore/mindspore/blob/master/mindspore/common/tensor.py>_`.
    ``bool_``                       布尔型，值为 ``True`` 或者 ``False`` 。
    ``int_``                        整数标量。
    ``uint``                        无符号整数标量。
    ``float_``                      浮点标量。
    ``complex``                     复数标量。
    ``number``                      数值型, 包括 ``int_`` , ``uint`` , ``float_`` , ``complex`` 和 ``bool_``。
    ``list_``                       由 ``tensor`` 构造的列表，例如 ``List[T0,T1,...,Tn]`` ，其中元素 ``Ti`` 可以是不同的类型。
    ``tuple_``                      由 ``tensor`` 构造的元组，例如 ``Tuple[T0,T1,...,Tn]`` ，其中元素 ``Ti`` 可以是不同的类型。
    ``function``                    函数类型。 两种返回方式，当function不是None时，直接返回Func，另一种当function为None时返回Func(参数: List[T0,T1,...,Tn], 返回值: T)。
    ``type_type``                   类型的类型定义。
    ``type_none``                   没有匹配的返回类型，对应 Python 中的 ``type(None)``。
    ``symbolic_key``                在 ``env_type`` 中用作变量的键的变量的值。
    ``env_type``                    用于存储函数的自由变量的梯度，其中键是自由变量节点的``symbolic_key``，值是梯度。
    ============================   =================
  
  * **树形拓扑**
  
    以上定义的数据类型遵从如下的树形拓扑结构：
  
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
            │   ├─── float_
            │   │    ├─── float16
            │   │    ├─── float32
            │   │    └─── float64
            │   └─── complex
            │       ├─── complex64
            │       └─── complex128
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
            ├─── type_type
            ├─── type_none
            ├─── symbolic_key
            └─── env_type
  
.. automodule:: mindspore
    :members:

    .. py:method:: mindspore.run_check()

        提供了便捷的API用以查询MindSpore的安装是否成功。

        样例:

        .. code-block::

              >>> import mindspore
              >>> mindspore.run_check()
              MindSpore version: xxx
              The result of multiplication calculation is correct, MindSpore has been installed successfully!

    .. py:method:: mindspore.dtype_to_nptype(type_)

        将MindSpore dtype转换成numpy数据类型。

        参数：

            **type_**(mindspore.dtype) – MindSpore中的dtype。

        返回:

            numpy的数据类型。

    .. py:method:: mindspore.issubclass_(type_, dtype)

        判断*type_*是否为*dtype*的子类。

        参数：

            - **type_**(mindspore.dtype) – MindSpore中的目标dtype。
            - **dtype**(mindspore.dtype) – dtype的比较对象。

        返回:

            布尔值，True或False。

    .. py:method:: mindspore.dtype_to_pytype(type_)

        将MindSpore dtype转换为python数据类型。

        参数：

            **type_**(mindspore.dtype) – MindSpore中的dtype。

        返回:

            python的数据类型。


    .. py:method:: mindspore.pytype_to_dtype(obj)

        将python数据类型转换为MindSpore数据类型。

        参数：

            **obj**(type) – python数据对象。

        返回:

            MindSpore的数据类型。

    .. py:method:: mindspore.get_py_obj_dtype(obj)

        将python数据类型转换为MindSpore数据类型。

        参数：

            **obj**(type) – python数据对象，或在python环境中定义的变量。

        返回:

            MindSpore的数据类型。

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None)

    张量被用于数据的存储。

    MindSpore中的张量继承于C++中的张量对象。部分函数由C++编写，部分函数由Python编写。

    参数：

    - **input_data**(Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]) – 输入的张量数据。
    - **dtype**(mindspore.dtype) – 张量的数据类型需为*mindSpore.dtype*中的None，bool或者数值型。这个参数是用于定义输出张量的数据类型。如果该参数为None，则输出张量的数据类型和*input_data*一致，默认参数：None。
    - **shape**(Union[tuple, list, int]) – 该参数为输出张量的形式，可由一列整数、一个元组、或一个整数表示。如果输入张量的形式已被定义，则无需设置该参数。
    - **init**(Initializer) – 'init'数据的相关信息。'init'被用于在并行模式中延迟初始化，通常来说，不推荐在其他条件下使用该接口初始化参数，只有当调用*Tensor.init_data* API用以转换张量数据时，才会使用'init'接口来初始化参数。

    输出:

    张量。如果*dtype*和数据形式未被设置，将会返回与输入张量相同的*dtype*和形式；否则，输出张量的*dtype*和形式将是用户指定的设置。

    样例:

    .. code-block::

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.common.initializer import One
        >>> # initialize a tensor with input data
        >>> t1 = Tensor(np.zeros([1, 2, 3]), ms.float32)
        >>> assert isinstance(t1, Tensor)
        >>> assert t1.shape == (1, 2, 3) 
        >>> assert t1.dtype == ms.float32

        >>> # initialize a tensor with a float scalar
        >>> t2 = Tensor(0.1)
        >>> assert isinstance(t2, Tensor)
        >>> assert t2.dtype == ms.float64

        >>> # initialize a tensor with init
        >>> t3 = Tensor(shape = (1, 3), dtype=ms.float32, init=One())
        >>> assert isinstance(t3, Tensor)
        >>> assert t3.shape == (1, 3)
        >>> assert t3.dtype == ms.float32

    .. py:property:: T
  
        返回转置后的张量。

    .. py:method:: abs()

        按元素返回绝对值。

        返回:

        张量，值为按元素返回的绝对值。

        支持平台:

        `Ascend` `GPU` `CPU`

        样例:

        .. code-block::

            >>> from mindspore import Tensor
            >>> a = Tensor([1.1, -2.1]).astype("float32")
            >>> output = a.abs()
            print(output)





