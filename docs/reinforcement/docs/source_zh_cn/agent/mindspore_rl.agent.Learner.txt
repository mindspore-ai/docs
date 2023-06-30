
.. py:class:: mindspore_rl.agent.Learner

    Learner的基类。通过输入的经验数据，计算并更新自生的网络。

    .. py:method:: learn(experience)

        learn 方法的接口。 `learn` 方法的行为取决于用户的实现。通常，它接受来自重放缓存中的 `samples` 或其他Tensors，并计算用于更新网络的损失。

        参数：
            - **experience** (tuple(Tensor)) - 缓存中的经验数据。

        返回：
            - **results** (tuple(Tensor)) - 更新权重后输出的结果。
