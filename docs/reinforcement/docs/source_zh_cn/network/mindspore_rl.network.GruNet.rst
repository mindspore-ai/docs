
.. py:class:: mindspore_rl.network.GruNet(input_size, hidden_size, weight_init='normal', num_layers=1, has_bias=True, batch_first=False, dropout=0.0, bidirectional=False)

    GRU (门控递归单元)层。
    将GRU层应用于输入。
    有关详细信息，请参见：:class:`mindspore.nn.GRU`。

    参数：
        - **input_size** (int) - 输入的特征数。
        - **hidden_size** (int) - 隐藏层的特征数量。
        - **weight_init** (str/Initializer) - 初始化方法，如normal、uniform。默认值： 'normal'。
        - **num_layers** (int) - GRU层的数量。默认值： 1。
        - **has_bias** (bool) - cell中是否有偏置。默认值： True。
        - **batch_first** (bool) - 指定输入 `x` 的第一个维度是否为批处理大小。默认值： False。
        - **dropout** (float) - 如果不是0.0, 则在除最后一层外的每个GRU层的输出上附加 `Dropout` 层。默认值： 0.0。取值范围 [0.0, 1.0)。
        - **bidirectional** (bool) - 指定它是否为双向GRU，如果bidirectional=True则为双向，否则为单向。默认值： False。

    输入：
        - **x_in** (Tensor) - 数据类型为mindspore.float32和shape为(seq_len, batch_size, `input_size`)或(batch_size, seq_len, `input_size`)的Tensor。
        - **h_in** (Tensor) - 数据类型为mindspore.float32和shape为(num_directions * `num_layers`, batch_size, `hidden_size`)的Tensor。`h_in` 的数据类型必须和 `x_in` 一致。

    输出：
        元组，包含(`x_out`, `h_out`)。

        - **x_out** (Tensor) - shape为(seq_len, batch_size, num_directions * `hidden_size`) 或(batch_size, seq_len, num_directions * `hidden_size`)的Tensor。
        - **h_out** (Tensor) - shape为(num_directions * `num_layers`, batch_size, `hidden_size`)的Tensor。

    .. py:method:: construct(x_in, h_in)

        gru网络的正向输出。

        参数：
            - **x_in** (Tensor) - 数据类型为mindspore.float32和shape为(seq_len, batch_size, `input_size`)或(batch_size, seq_len, `input_size`)的Tensor。
            - **h_in** (Tensor) - 数据类型为mindspore.float32和shape为(num_directions * `num_layers`, batch_size, `hidden_size`)的Tensor。`h_in` 的数据类型必须和 `x_in` 一致。

        返回：
            - **x_out** (Tensor) - shape为(seq_len, batch_size, num_directions * `hidden_size`) 或(batch_size, seq_len, num_directions * `hidden_size`)的Tensor。
            - **h_out** (Tensor) - shape为(num_directions * `num_layers`, batch_size, `hidden_size`)的Tensor。
