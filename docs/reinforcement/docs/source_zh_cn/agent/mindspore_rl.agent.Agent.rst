
.. py:class:: mindspore_rl.agent.Agent(actors, learner)

    Agent的基类。作为智能体的定义，由Actor和Learner构成。具备基本的act和learn功能用于和环境交互和自我更新。

    参数：
        - **actors** (Actor) - Actor 实例。
        - **learner** (Learner) - learner 实例。

    .. py:method:: act(phase, params)

        act 方法接收一个枚举值和观察数据或计算动作期间所需的数据。它将返回一组包含新观察数据或其他经验的输出。此接口中，Agent将与环境交互。

        参数：
            - **phase** (enum) - 一个int型的枚举值，用于初始化、收集或评估的阶段。
            - **params** (tuple(Tensor)) - 作为输入的张量元组，用于计算动作。

        返回：
            - **observation** (tuple(Tensor)) - 作为输出的张量元组，用于生成经验数据。

    .. py:method:: get_action(phase, params)

        get_action 方法接收一个枚举值和观察数据或计算动作期间所需的数据。它将返回一组包含动作和其他数据的输出。此接口中，Agent不与环境交互。

        参数：
            - **phase** (enum) - 一个int型的枚举值，用于初始化、收集、评估或者其他用户定义的阶段。
            - **params** (tuple(Tensor)) - 作为输入的张量元组，用于计算动作。

        返回：
            - **action** (tuple(Tensor)) - 作为输出的张量元组，包含动作和其他所需数据的张量。
    
    .. py:method:: learn(experience)

        learn 方法接收一组经验数据作为输入，以计算损失并更新权重。

        参数：
            - **experience** (tuple(Tensor)) - 经验的张量状态元组。

        返回：
            - **results** (tuple(Tensor)) - 更新权重后输出的结果。
