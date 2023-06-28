
.. py:class:: mindspore_rl.utils.DiscountedReturn(gamma, need_bprop)

    计算折扣回报。

    设折扣回报为 :math:`G`，折扣系数为 :math:`\gamma`，奖励为 :math:`R`，时间步 :math:`t`，最大时间步 :math:`N`。
    则 :math:`G_{t} = \Sigma_{t=0}^N{\gamma^tR_{t+1}}`。

    对于奖励序列包含多个episode的情况， :math:`done` 用来标识episode边界， :math:`last\_state\_value` 表示最后一个epsode的最后一个step的价值。

    参数：
        - **gamma** (float) - 折扣系数。
        - **need_bprop** (bool) - 是否需要计算discounted return的反向

    输入：
        - **reward** (Tensor) - 包含多个episode的奖励序列。 张量的维度 :math:`(Timestep, Batch, ...)`。
        - **done** (Tensor) - Episode结束标识。 张量维度 :math:`(Timestep, Batch)`。
        - **last_state_value** (Tensor) - 表示最后一个epsode的最后一个step的价值， 张量的维度 :math:`(Batch, ...)`。

    返回：
        折扣回报。
