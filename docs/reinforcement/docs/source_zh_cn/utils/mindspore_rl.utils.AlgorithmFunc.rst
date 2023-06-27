
.. py:class:: mindspore_rl.utils.AlgorithmFunc

    这是MCTS中用户用来自定义传入算法的基类。用户需要继承这个基类并且按照提供的输入输出实现对应的方法。

    .. py:method:: calculate_prior(new_state, legal_action)

        calculate_prior的功能是计算输入合法动作的先验概率。
        
        参数：
            - **new_state** (mindspore.float32) - 环境的状态。
            - **legal_action** (mindspore.int32) - 环境输出的合法动作。

        返回：
            - **prior** (mindspore.float32) - 每个动作的先验概率。


    .. py:method:: simulation(new_state)

        simulation的功能是计算输入状态的奖励（评估价值）。
        
        参数：
            - **new_state** (mindspore.float32) - 环境的状态。

        返回：
            - **rewards** (mindspore.float32) - simulation的结果。