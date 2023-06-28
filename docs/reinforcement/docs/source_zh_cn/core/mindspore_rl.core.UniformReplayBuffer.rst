
.. py:class:: mindspore_rl.core.UniformReplayBuffer(batch_size, capacity, shapes, types)

    重放缓存类。重放缓存区中存放来自环境的经验数据。在该类中，每个元素都是一组Tensor。因此，ReplayBuffer类的构造函数将每个Tensor的形状和类型作为参数。

    参数：
        - **batch_size** (int) - 从缓存区采样的batch大小。
        - **capacity** (int) - 缓存区的大小。
        - **shapes** (list[int]) - 缓存区中每个元素对应的Tensor shape列表。
        - **types** (list[mindspore.dtype]) - 缓存区中每个元素对应的Tensor dtype列表。

    .. py:method:: full()

        检查缓存区是否已满。

        返回：
            - **Full** (bool) - 缓存区已满返回True，否则返回False。

    .. py:method:: get_item(index)

        从缓存区的指定位置取出元素。

        参数：
            - **index** (int) - 元素的索引。

        返回：
            - **element** (list[Tensor]) - 返回指定位置的元素。

    .. py:method:: insert(exp)

        将元素插入缓存区。如果缓存区已满，则将使用先进先出的策略替换缓存区的元素。

        参数：
            - **exp** (list[Tensor]) - 插入的Tensor组，需要符合缓存初始化时的shape和type。

        返回：
            - **element** (list[Tensor]) - 返回插入数据后的缓存区。

    .. py:method:: reset()

        重置缓存区，将count值置零。

        返回：
            - **success** (bool) - 重置是否成功。

    .. py:method:: sample()

        缓存区采样，随机地选择一组元素并输出。

        返回：
            - **data** (Tuple(Tensor)) - 一组从缓存区随机采样出的元素。

    .. py:method:: size()

        返回缓存区的大小。

        返回：
            - **size** (int) - 缓存区的元素个数。
