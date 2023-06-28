
.. py:class:: mindspore_rl.agent.Actor

    所有Actor的基类。Actor 是一个用来和环境交互并产生数据的类。

    .. py:method:: act(phase, params)

        act 方法接收一个枚举值和观察数据或计算动作期间所需的数据。它将返回一组包含新观察数据或其他经验的输出。此接口将与环境交互。

        参数：
            - **phase** (enum) - 一个int型的枚举值，用于初始化、收集、评估或其他用户定义的阶段。
            - **params** (tuple(Tensor)) - 作为输入的张量元组，用于计算动作。

        返回：
            - **observation** (tuple(Tensor)) - 作为输出的张量元组，用于生成经验数据。

    .. py:method:: get_action(phase, params)

        get_action 是用来获得动作的方法。用户需要根据算法重载此函数。但该函数入参需为phase和params。此接口不会与环境交互。

        参数：
            - **phase** (enum) - 一个int型的枚举值，用于初始化、收集、评估或者其他用户定义的阶段。
            - **params** (tuple(Tensor)) - 作为输入的张量元组，用于计算动作。

        返回：
            - **action** (tuple(Tensor)) - 作为输出的张量元组，包含动作和其他所需数据的张量。
