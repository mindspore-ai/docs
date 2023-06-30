
.. py:class:: mindspore_rl.policy.Policy

    策略的虚基类。在调用模型之前，应该重写此类。

    .. py:method:: construct(*inputs, **kwargs)

        构造函数接口。由用户继承使用，参数可参考 `EpsilonGreedyPolicy`， `RandomPolicy` 等。

        参数：
            - **inputs** - 取决于用户的定义。
            - **kwargs** - 取决于用户的定义。

        返回：
            取决于用户的定义。通常返回一个动作值或者动作的概率分布。
