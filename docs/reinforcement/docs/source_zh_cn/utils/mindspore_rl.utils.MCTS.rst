
.. py:class:: mindspore_rl.utils.MCTS(env, tree_type, node_type, root_player, customized_func, device, args, has_init_reward=False, max_action=-1.0, max_iteration=1000)
    
    蒙特卡洛树搜索（MCTS）是一种通用搜索决策算法，在棋类游戏（如围棋，国际象棋）中效果尤为显著。MCTS在2006年被首次提出。一个通用的MCTS会有以下四个阶段：

    1. 选择（Selection） - 根据选择策略（如UCT, RAVE, AMAF等）选择下一个节点。
    2. 扩展（Expansion） - 除非搜索达到了终止节点，新的子节点都会被添加到选择阶段达到的叶节点。
    3. 模拟（Simulation） - 使用一个算法（随机，神经网络或者其他算法）去获得当前状态的回报。
    4. 反向传播（Backpropagation） - 把模拟计算出的回报传播给所有经过的节点。

    随着时间的推移，MCTS中的四步都更新迭代。AlphaGo中就在MCTS中引入了神经网络，使得MCTS更加强大。

    本MCTS类由MindSpore算子组成。用户可以直接使用提供个MCTS算法，或者通过继承C++中的MonteCarloTreeNode去开发自己的MCTS算法。

    参数：
        - **env** (Environment) - 必须是Environment的子类。
        - **tree_type** (str) - 树类型的名字。
        - **node_type** (str) - 节点类型的名字。
        - **root_player** (float) - 根节点的玩家，数值需要小于总玩家数。
        - **customized_func** (AlgorithmFunc) - 算法相关的类。更多信息请参考AlgorithmFunc的文档。
        - **device** (str) - 运行MCTS的设备['CPU', 'GPU']，Ascend当前不支持。
        - **args** (Tensor) - 在MctsCreation中传入的常量值。请参考以下表格根据算法传入输入值。这里传入的值不会在'restore_tree_data'方法中被重置。
        - **has_init_reward** (bool) - 是否把奖励在初始化时传给节点。默认：False。
        - **max_action** (float) - 环境的最大动作。当max_action是-1.0时，环境的step函数只会获得最后一个动作，否则环境的step函数会获得所有动作。默认：-1.0.
        - **max_iteration** (int) - 最多的训练迭代次数。默认：1000.

        +------------------------------+-----------------+-----------------------------+--------------------------+
        |  MCTS树类型                  |  MCTS节点类型   |  配置参数                   |  备注                    |
        +==============================+=================+=============================+==========================+
        |  CPUCommon                   |  CPUVanilla     |  UCT常量                    |  UCT常量被使用在Selection|
        +------------------------------+-----------------+-----------------------------+  阶段，去计算UCT值。     |
        |  GPUCommon                   |  GPUVanilla     |  UCT常量                    |                          |
        +------------------------------+-----------------+-----------------------------+--------------------------+

    .. py:method:: destroy(handle)

        销毁当前这棵树。请在算法结束或不再需要这棵树时调用。
        
        参数：
            - **handle** (mindspore.int64) - 独有的蒙特卡洛树句柄。

        返回：
            - **action** (mindspore.bool\_) - 是否成功重置。

    .. py:method:: mcts_search(*args)

        mcts_search是MCTS中的主要方法。调用此方法会返回当前状态下的最优动作。
        
        参数：
            - **args** (Tensor) - 在迭代中会更新的变量，并且在调用'restore_tree_data'中会重置。输入值需要和算法对应。

        返回：
            - **action** (mindspore.int32) - 蒙特卡洛树搜索返回的动作。
            - **handle** (mindspore.int64) - 独有的蒙特卡洛树句柄。

    .. py:method:: restore_tree_data(handle)

        restore_tree_data会重置树中的所有信息，回到只有根节点的状态。
        
        参数：
            - **handle** (mindspore.int64) - 独有的蒙特卡洛树句柄。

        返回：
            - **action** (mindspore.bool\_) - 是否成功重置。
