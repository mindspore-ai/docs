
.. py:class:: mindspore_rl.network.FullyConnectedNet(input_size, hidden_size, output_size, compute_type=mstype.float32)

    一个基本的全连接神经网络。

    参数：
        - **input_size** (int) - 输入的数量。
        - **hidden_size** (int) - 隐藏层的数量。
        - **output_size** (int) - 输出大小的数量。
        - **compute_type** (mindspore.dtype) - 用于全连接层的数据类型。默认值： mindspore.float32。

    .. py:method:: construct(x)

        返回网络的输出。

        参数：
            - **x** (Tensor) - 网络的输入张量。

        返回：
            网络的输出。

.. py:class:: mindspore_rl.network.FullyConnectedLayers(fc_layer_params, dropout_layer_params=None, activation_fn=nn.ReLU(), weight_init='normal', bias_init='zeros')

    这是一个全连接层的模块。用户可以输入任意数量的fc_layer_params，然后该模块可以创建相应数量的全链接层。

    参数：
        - **fc_layer_params** (list[int]) - 全连接层输入和输出大小的值列表。例如，输入列表为[10，20，30]，模块将创建两个全连接层，
          其输入和输出大小分别为(10, 20)和(20,30)。fc_layer_params的长度应大于等于3。
        - **dropout_layer_params** (list[float]) - 丢弃率的列表。如果输入为[0.5, 0.3]，则在每个全连接层之后将创建两个丢弃层。
          dropout_layer_params的长度应小于fc_layer_params。 dropout_layer_params是个可选值。默认值： None。
        - **activation_fn** (Union[str, Cell, Primitive]) - 激活函数的实例。默认值： nn.ReLU()。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始化权重参数。类型与 `x` 相同。str的值代表
          `Initializer` 函数，如normal、uniform。默认值： 'normal'。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始化偏置参数。类型与 `x` 相同。str的值代表
          `Initializer` 函数，如normal、uniform。默认值： 'zeros'。

    输入：
        - **x** (Tensor) - Tensor的shape为 :math:`(*, fc\_layers\_params[0])`。

    输出：
        Tensor的shape为 :math:`(*, fc\_layers\_params[-1])`。

    .. py:method:: construct(x)

        返回网络的输出。

        参数：
            - **x** (Tensor) - Tensor的shape为 :math:`(*, fc\_layers\_params[0])`。

        返回：
            Tensor的shape为 :math:`(*, fc\_layers\_params[-1])`。
