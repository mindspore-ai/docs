.. py:class:: mindspore_rl.environment.TicTacToeEnvironment(params, env_id=0)

    井字棋是一款有名的纸笔游戏<https://en.wikipedia.org/wiki/Tic-tac-toe>。这个游戏的规则是两个玩家在一个3X3的格子上交互的画O和X。当三个相同的标记在水平，垂直或者对角线连成一条线时，对应的玩家将获得胜利。下图就是一个井字棋游戏的例子。

    +---+---+---+
    | o |   | x |
    +---+---+---+
    | x | o |   |
    +---+---+---+
    |   | x | o |
    +---+---+---+

    参数：
        - **params** (dict) - 字典包含TicTacToeEnvironment类中所需要的所有参数。
        - **env_id** (int) - 环境id，用于设置环境内种子。默认：0。

    .. py:method:: action_space
        :property:

        获取环境的动作空间。

        返回：
            Space，环境的动作空间。

    .. py:method:: calculate_rewards()

        返回当前状态的收益。

        返回：
            Tensor，表示当前状态收益。

    .. py:method:: config
        :property:

        获取环境的配置信息。

        返回：
            dict，一个包含环境信息的字典。

    .. py:method:: current_player()

        返回当前状态下，轮到哪个玩家。

        返回：
            Tensor，表示当前玩家。

    .. py:method:: done_space
        :property:

        获取环境的终止空间。

        返回：
            Space，环境的终止空间。

    .. py:method:: is_terminal()

        返回当前状态下，游戏是否已经终止。

        返回：
            当前状态下，游戏是否已经终止。

    .. py:method:: legal_action()

        返回当前状态的合法动作

        返回：
            Tensor，表示合法动作。

    .. py:method:: load(state)

        加载输入的状态。环境会根据输入的状态，更新当前的状态，合法动作和是否结束。

        参数：        
            - **state** (Tensor) - 输入的环境状态。

        返回：
            - **state** (Tensor) - 存档点的状态。
            - **reward** (Tensor) - 存档点的收益。
            - **done** (Tensor) - 是否在输入存档点时，游戏已经结束。

    .. py:method:: max_utility()

        返回井字棋游戏的最大收益。

        返回：
            Tensor，表示最大收益。

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

    .. py:method:: save()

        返回一个环境的副本。在井字棋游戏中不需要返回环境的副本，因此他会返回当前状态。

        返回：
            一个代表当前状态的Tensor。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一回合。

        参数：
            - **action** (Tensor) - 包含动作信息的Tensor。

        返回：
            - **state** (Tensor) - 输入动作后的环境返回的新状态。
            - **reward** (Tensor) - 输入动作后环境返回的奖励。
            - **done** (Tensor) - 输入动作后环境是否终止。

    .. py:method:: total_num_player()

        返回总玩家数量。

        返回：
            Tensor，表示总玩家数量。
