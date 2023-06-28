
.. py:class:: mindspore_rl.utils.TensorArray(dtype, element_shape, dynamic_size=True, size=0, name="TA")

    用来存Tensor的动态数组。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。

    参数：
        - **dtype** (mindspore.dtype) - 动态数组的数据类型。
        - **element_shape** (tuple(int)) - 动态数组中每个Tensor的shape。
        - **dynamic_size** (bool) - 如果是true，则该数组可以动态增长，否则为固定大小。默认：True。
        - **size** (int) - 如果 `dynamic_size=False` , 则 `size` 表示该数组的最大容量。
        - **name** (str) - 动态数组的名字。默认："TA"。

    .. py:method:: clear()

        清理创建的动态数组。仅重置该数组，清理数据和重置大小，保留数组实例。

        返回：
            True。

    .. py:method:: close()

        关闭动态数组。

        .. warning::
            - 一旦关闭了动态数组，每个属于该动态数组的方法都将失效。所有该数组中的资源也将被清除。如果该数组还将在别的地方使用，如下一个循环，请用 `clear` 代替。

        返回：
            True。

    .. py:method:: read(index)

        从动态数组的指定位置读Tensor。

        参数：
            - **index** ([int, mindspore.int64]) - 读取的位置。

        返回：
            Tensor, 指定位置的值。

    .. py:method:: size()

        动态数组的逻辑大小。

        返回：
            Tensor, 动态数组大小。

    .. py:method:: stack()

        堆叠动态数组中的Tensor为一个整体。

        返回：
            Tensor, 动态数组中的所有Tensor将堆叠成一个整体。

    .. py:method:: write(index, value)

        向动态数组的指定位置写入值（Tensor）。

        参数：
            - **index** ([int, mindspore.int64]) - 写入的位置。
            - **value** (Tensor) - 写入的Tensor。

        返回：
            True。