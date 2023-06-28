.. py:class:: mindspore_rl.environment.Space(feature_shape, dtype, low=None, high=None, batch_shape=None)

    包含环境动作/状态空间的类。

    参数：
        - **feature_shape** (Union[list(int), tuple(int), int]) - 批处理前的动作/状态的Shape。
        - **dtype** (np.dtype) - 动作/状态空间的数据类型。
        - **low** (int, float) - 动作/状态空间的下边界。默认：None。
        - **high** (int, float) - 动作/状态空间的上边界。默认：None。
        - **batch_shape** (Union[list(int), tuple(int), int]) - 矢量化的批量Shape。通常用于多环境和多智能体的场景。默认：None。

    .. py:method:: boundary
        :property:

        **返回：**

        当前空间的上下边界。

    .. py:method:: is_discrete
        :property:

        **返回：**

        是否为离散空间。

    .. py:method:: ms_dtype
        :property:

        **返回：**

        当前空间的MindSpore的数据类型。

    .. py:method:: np_dtype
        :property:

        **返回：**

        当前空间的Numpy的数据类型。

    .. py:method:: num_values
        :property:

        **返回：**

        当前空间可选动作的数量。

    .. py:method:: shape
        :property:

        **返回：**

        批处理后的Space的Shape。

    .. py:method:: sample()

        从当前Space里随机采样一个合法动作。

        返回：
            - **action** (Tensor) - 一个合法动作的Tensor。
