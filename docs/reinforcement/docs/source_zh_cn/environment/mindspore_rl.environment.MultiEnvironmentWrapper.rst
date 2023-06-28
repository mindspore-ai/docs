.. py:class:: mindspore_rl.environment.MultiEnvironmentWrapper(env_instance, num_proc=None)

    MultiEnvironmentWrapper是多环境场景下的包装器。用户实现自己的单环境类，并在配置文件中设置环境数量大于1时，框架将自动调用此类创建多环境。

    参数：
        - **env_instance** (list[Environment]) - 包含环境实例（继承Environment类）的List。
        - **num_proc** (int) - 在和环境交互时使用的进程数量。默认值： None。

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            Space，环境的动作空间。

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (bool) - 是否成功释放资源。

    .. py:method:: config
        :property:

        获取环境的配置信息。

        返回：
            dict，一个包含环境信息的字典。

    .. py:method:: done_space
        :property:

        获取环境的终止空间。

        返回：
            Space，环境的终止空间。

    .. py:method:: observation_space
        :property:

        获取环境的状态空间。

        返回：
            Space，返回环境的状态空间。

    .. py:method:: render()

        渲染环境，仅支持PyNative模式。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每一局游戏开始时使用，并返回环境的初始状态值。

        返回：
            表示环境初始状态的Tensor List。

    .. py:method:: reward_space
        :property:

        获取环境的状态空间。

        返回：
            Space，环境的奖励空间。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Tensor) - 包含动作信息的Tensor。

        返回：        
            - **state** (list(Tensor)) - 输入动作后的环境返回的新状态List。
            - **reward** (list(Tensor)) - 输入动作后环境返回的奖励List。
            - **done** (list(Tensor)) - 输入动作后环境是否终止的List。

