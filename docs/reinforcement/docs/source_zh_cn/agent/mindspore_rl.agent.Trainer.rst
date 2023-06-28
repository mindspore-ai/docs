
.. py:class:: mindspore_rl.agent.Trainer(msrl)

    Trainer的基类。是一个流程类，提供训练的基本模式。

    参数：
        - **msrl** (MSRL) - 函数句柄。

    .. py:method:: evaluate

        在训练中用于评估的评估方法。

    .. py:method:: load_and_eval(ckpt_path=None)

        离线评估的方法。必须提供一个checkpoint。

        参数：
            - **ckpt_path** (string) - 需要加载到网络的checkpoint文件。默认值：None。

    .. py:method:: train(episodes, callbacks=None, ckpt_path=None)

        train 方法中提供一个标准的训练流程，包含整个循环和回调。用户可根据需要自行继承或覆写。

        参数：
            - **episodes** (int) - 训练回合数。
            - **callbacks** (Optional[list[Callback]]) - 回调函数的列表。默认值：None。
            - **ckpt_path** (Optional[str]) - 要初始化或重加载的网络文件路径。默认值：None。

    .. py:method:: train_one_episode

        在训练中，训练一个回合的接口。该函数的输出必须按顺序限制为 `loss, rewards, steps, [Optional]others`。

    .. py:method:: trainable_variables

        用于保存至checkpoint的变量。
