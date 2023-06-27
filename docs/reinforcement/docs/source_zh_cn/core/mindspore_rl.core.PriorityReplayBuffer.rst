
.. py:class:: mindspore_rl.core.PriorityReplayBuffer(alpha, capacity, sample_size, shapes, dtypes, seed0=0, seed1=0)

    优先级经验回放缓存，用于深度Q学习存储经验数据。
    该算法在 `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_ 中提出。
    与普通的经验回放缓存相同，它允许强化学习智能体记住和重用过去的经验。此外，它更频繁的回放重要的transition，提高样本效率。

    参数：
        - **alpha** (float) - 控制优先级程度的参数。0表示均匀采样，1表示优先级采样。
        - **capacity** (int) - 缓存的容量。
        - **sample_size** (int) - 从缓存采样的大小
        - **shapes** (list[int]) - 缓存区中张量维度列表。
        - **types** (list[mindspore.dtype]) - 缓存区张量数据类型列表。
        - **seed0** (int) - 随机数种子0值。默认值：0。
        - **seed1** (int) - 随机数种子1值。默认值：0。

    .. py:method:: destroy()

        销毁经验回放缓存。

        返回：
            - **handle** (Tensor) - 优先级经验回放缓存句柄，数据和维度分别是int64和（1,）。

    .. py:method:: push(*transition)

        将transition推送到缓存区。如果缓存区已满，则覆盖最早的数据。

        参数：
            - **transition** (List[Tensor]) - 与初始化的shapes和dtypes匹配的张量列表。

        返回：
            - **handle** (Tensor) - 优先级经验回放缓存句柄，数据和维度分别是int64和（1,）。


    .. py:method:: sample(beta)

        从缓存区中采样一批transition。

        参数：
            - **beta** (float) - 控制采样校正程度的参数。0表示不校正，1表示完全校正。

        返回：
            - **indices** (Tensor) - transition在缓存区中的索引。
            - **weights** (Tensor) - 用于校正采样偏差的权重。
            - **transition** -  采样得到的transition。

    .. py:method:: update_priorities(indices, priorities)

        更新transition的优先级。

        参数：
            - **indices** (Tensor) - transition在缓存区中的索引。
            - **priorities** (Tensor) - transition优先级。

        返回：
            - **handle** (Tensor) - 优先级经验回放缓存句柄，数据和维度分别是int64和（1,）。
