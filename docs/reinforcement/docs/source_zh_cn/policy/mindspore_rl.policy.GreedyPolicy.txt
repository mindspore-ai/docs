
.. py:class:: mindspore_rl.policy.GreedyPolicy(input_network)

    基于给定的贪婪策略生成采样动作。

    参数：
        - **input_network** (Cell) - 用于按输入状态产生动作的网络。

    .. py:method:: construct(state)

        返回最佳动作。

        参数：
            - **state** (Tensor) - 网络的输入状态Tensor。

        返回：
            action_max，输出最佳动作。
