.. py:class:: mindspore_rl.environment.StarCraft2Environment(params, env_id=0)

    StarCraft2Environment是一个SMAC的包装器。SMAC是WhiRL的一个基于暴雪星际争霸2这个战略游戏开发的用于多智能体强化学习（MARL）在合作场景的环境。SMAC通过使用暴雪星际争霸2的机器学习API和DeepMind的PySC2提供了易用的界面方便智能体与星际争霸2的交互来获得环境的状态和合法的动作。不像PySC2，SMAC专注于去中心的细微操控场景，这种场景下游戏中的每个单位都会被一个独立的RL智能体操控。更多的信息请查阅官方的SMAC官方的GitHub：
    <https://github.com/oxwhirl/smac>。

    参数：
        - **params** (dict) - 字典包含StarCraft2Environment类中所需要的所有参数。

          +------------------------------+---------------------------------------------------+
          |  配置参数                    |  备注                                             |
          +==============================+===================================================+
          |  sc2_args                    |  一个用于创建SMAC实例的字典包含一些SMAC需要的key值|
          |                              |  如map_name. 详细配置信息请查看官方GitHub。       |
          +------------------------------+---------------------------------------------------+

        - **env_id** (int) - 环境id，用于设置环境内种子。

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            Space，返回环境的动作空间。

    .. py:method:: close

        关闭环境以释放环境资源

        返回：
            - **Success** (bool) - 是否成功释放资源。

    .. py:method:: config
        :property:

        获取环境的配置信息。

        返回：
            dict，返回一个包含环境信息的字典。

    .. py:method:: done_space
        :property:

        获取环境的终止空间。

        返回：
            Space，返回环境的终止空间。

    .. py:method:: get_step_info()

        在与环境交互后，获得环境的信息。

        返回：
            - **battle_won** (Tensor) - 是否这局游戏取得胜利。
            - **dead_allies** (Tensor) - 己方单位阵亡数量。
            - **dead_enemies** (Tensor) - 敌方单位阵亡数量。

    .. py:method:: observation_space
        :property:

        获取环境的状态空间。

        返回：
            返回环境的状态空间。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每一局游戏开始时使用，并返回环境的初始状态值，全局状态以及新的合法动作。

        返回：
            tuple，包含了环境的初始状态值，全局状态以及新的合法动作这几个Tensor。

    .. py:method:: reward_space
        :property:

        获取环境的奖励空间。

        返回：
            Space，返回环境的奖励空间。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Tensor) - 包含动作信息的Tensor。

        返回：
            - **state** (Tensor) - 输入动作后的环境返回的新状态。
            - **reward** (Tensor) - 输入动作后环境返回的奖励。
            - **done** (Tensor) - 输入动作后环境是否终止。
            - **global_obs** (Tensor) - 输入动作后环境返回的新的全局状态。
            - **avail_actions** (Tensor) - 输入动作后环境返回的新的合法动作。
