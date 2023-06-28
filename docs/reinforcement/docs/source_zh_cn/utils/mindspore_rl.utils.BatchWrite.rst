
.. py:class:: mindspore_rl.utils.BatchWrite

    写一个list的参数覆盖到目标值。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。

    .. py:method:: construct(dst, src)

        将 `src` 中的参数覆盖到 `dst` 。

        参数：
            - **dst** (tuple(Parameters)) - 目标位置的参数列表。
            - **src** (tuple(Parameters)) - 源位置的参数列表。

        返回：
            True。

.. py:class:: mindspore_rl.utils.BatchRead

    读一个list的参数覆盖到目标值。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。

    .. py:method:: construct(dst, src)

        读取 `src` 中的参数覆盖到 `dst` 。

        参数：
            - **dst** (tuple(Parameters)) - 目标位置的参数列表。
            - **src** (tuple(Parameters)) - 源位置的参数列表。

        返回：
            True。
