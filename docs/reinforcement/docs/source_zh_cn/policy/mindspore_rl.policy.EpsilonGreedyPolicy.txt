
.. py:class:: mindspore_rl.policy.EpsilonGreedyPolicy(input_network, size, epsi_high, epsi_low, decay, action_space_dim)

    基于给定的epsilon-greedy策略生成采样动作。

    参数：
        - **input_network** (Cell) - 返回策略动作的输入网络。
        - **size** (int) - epsilon的shape。
        - **epsi_high** (float) - 探索的上限epsilon值，介于[0, 1]。
        - **epsi_low** (float) - 探索的下限epsilon值，介于[0, epsi_high]。
        - **decay** (float) - epsilon的衰减系数。
        - **action_space_dim** (int) - 动作空间的维度。

    .. py:method:: construct(state, step)

        构造函数接口。

        参数：
            - **state** (Tensor) - 网络的输入Tensor。
            - **step** (Tensor) - 当前step, 影响epsilon的衰减。

        返回：
            输出动作。
