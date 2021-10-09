mindspore.Tensor
================

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None)

    张量被用于数据的存储。

    MindSpore中的张量继承于C++中的张量对象。部分函数由C++编写，部分函数由Python编写。

    **参数：**

        - **input_data** (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]) – 输入的张量数据。
        - **dtype** (mindspore.dtype) – 张量的数据类型需为 *mindSpore.dtype* 中的None，bool或者数值型。这个参数是用于定义输出张量的数据类型。如果该参数为None，则输出张量的数据类型和 *input_data* 一致，默认参数：None。
        - **shape** (Union[tuple, list, int]) – 该参数为输出张量的形式，可由一列整数、一个元组、或一个整数表示。如果输入张量的形式已被定义，则无需设置该参数。
        - **init** (Initializer) – `init` 数据的相关信息。`init` 被用于在并行模式中延迟初始化，通常来说，不推荐在其他条件下使用该接口初始化参数，只有当调用 *Tensor.init_data* API用以转换张量数据时，才会使用 `init` 接口来初始化参数。

    **输出：**

        张量。如果 *dtype* 和数据形式未被设置，将会返回与输入张量相同的 *dtype* 和形式；否则，输出张量的 *dtype* 和形式将是用户指定的设置。

    **样例：**

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

        **返回：**

            张量，值为按元素返回的绝对值。

        **支持平台：**

            ``Ascend`` ``GPU`` ``CPU``

        **样例：**

            .. code-block::

                >>> from mindspore import Tensor
                >>> a = Tensor([1.1, -2.1]).astype("float32")
                >>> output = a.abs()
                print(output)