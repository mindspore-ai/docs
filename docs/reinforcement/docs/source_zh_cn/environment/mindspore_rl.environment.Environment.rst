.. py:class:: mindspore_rl.environment.Environment

    环境的虚基类。在调用此类之前，请重写其中的方法。

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            返回环境的动作空间。

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (bool) - 是否成功释放资源。

    .. py:method:: config
        :property:

        获取环境的配置信息。

        返回：
            返回一个包含环境信息的字典。

    .. py:method:: done_space
        :property:

        获取环境的终止空间。

        返回：
            返回环境的终止空间。

    .. py:method:: observation_space
        :property:

        获取环境的状态空间。

        返回：
            返回环境的状态空间。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每一局游戏开始时使用，并返回环境的初始状态值以及其reset方法初始信息。

        返回：
            表示环境初始状态的Tensor或者Tuple包含初始信息，，如新的状态，动作，奖励等。

    .. py:method:: reward_space
        :property:

        获取环境的状态空间。

        返回：
            返回环境的奖励空间。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Tensor) - 包含动作信息的Tensor。

        返回：
            tuple，包含和环境交互后的信息。
