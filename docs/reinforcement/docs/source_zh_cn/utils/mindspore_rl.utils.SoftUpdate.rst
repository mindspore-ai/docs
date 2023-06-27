
.. py:class:: mindspore_rl.utils.SoftUpdate(factor, update_interval, behavior_params, target_params)

    采用滑动凭据方式更新目标网络的参数。

    设目标网络参数为 :math:`target\_param`，行为网络参数为 :math:`behavior\_param`，
    滑动平均系数为 :math:`factor`。
    则 :math:`target\_param = (1. - factor) * behavior\_param + factor * target\_param`。

    参数：
        - **factor** (float) - 滑动平均系数，范围[0, 1]。
        - **update_interval** (int) - 目标网络参数更新间隔。
        - **behavior_params** (list) - 行为网络参数列表。
        - **target_params** (list) - 目标网络参数列表。
