
.. py:class:: mindspore_rl.core.Session(alg_config, deploy_config=None)

    Session是一个用于运行MindSpore RL算法的类。

    参数：
        - **alg_config** (dict) - 算法的配置或算法的部署配置。
        - **deploy_config** (dict) - 分布式的部署配置。更多算法配置的详细信息，请看 
          https://www.mindspore.cn/reinforcement/docs/zh-CN/r0.5/custom_config_info.html

    .. py:method:: run(class_type=None, is_train=True, episode=0, duration=0, params=None, callbacks=None)

        执行强化学习算法。

        参数：
            - **class_type** (Trainer) - 算法的trainer类的类型。默认值： None。
            - **is_train** (bool) - 在训练或推理中执行算法，True为训练，False为推理。默认值： True。
            - **episode** (int) - 训练的回合数。默认值： 0。
            - **duration** (int) - 每回合的步数。默认值： 0。
            - **params** (dict) - 算法特定的训练参数。默认值： None。
            - **callbacks** (list[Callback]) - 回调列表。默认值： None。
