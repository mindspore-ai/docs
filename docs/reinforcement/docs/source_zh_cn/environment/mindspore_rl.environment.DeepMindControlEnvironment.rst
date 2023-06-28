.. py:class:: mindspore_rl.environment.DeepMindControlEnvironment(params, env_id=0)
    
    DeepMindControlEnvironment将DeepMind Control Suite(DMC)通过MindSpore算子再次封装。它用于基于物理的模拟和强化学习环境，使用MUJOCO。

    参数：
        - **params** (dict) - 字典包含DeepMindControlEnvironment类中所需要的所有参数。

        +------------------------------+----------------------------+
        |  配置参数                    |          备注              |
        +==============================+============================+
        |  env_name                    |  DMC内游戏的名字           |
        +------------------------------+----------------------------+
        |  seed                        |  DMC内使用的随机种子       |
        +------------------------------+----------------------------+
        |  camera                      |  在渲染中使用的camera位置  |
        +------------------------------+----------------------------+
        |  action_repeat               |  同一个动作和环境交互几次  |
        +------------------------------+----------------------------+
        |  normalize_action            |  是否需要归一化输入动作    |
        +------------------------------+----------------------------+
        |  img_size                    |  渲染图像的大小            |
        +------------------------------+----------------------------+
        
        - **env_id** (int) - 环境id，用于设置环境内种子。默认：0。

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
            Space，环境的状态空间。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每一局游戏开始时使用，并返回环境的初始状态值。

        返回：
            Tensor，表示环境初始状态。

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
            - **state** (Tensor) - 输入动作后的环境返回的新状态。
            - **reward** (Tensor) - 输入动作后环境返回的奖励。
            - **done** (Tensor) - 输入动作后环境是否终止。
            - **discount** (Tensor) - 环境对于当前状态返回的折扣
