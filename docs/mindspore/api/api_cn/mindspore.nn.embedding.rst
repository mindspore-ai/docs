mindspore.nn.Embedding
=======================

.. py:class:: mindspore.nn.Embedding(vocab_size, embedding_size, use_one_hot=False, embedding_table="normal", dtype=mstype.float32, padding_idx=None)

   一个简单的查找表，保存固定字典和大小的嵌入（embeddings）。

   该模块常被用做存储词嵌入，并使用索引检索它们。该模块的输入是一个索引列表，输出是对应的词嵌入。

   .. note:: 当 `use_one_hot` 等于True时，x的类型必须是mindpore.int32。

   **参数** ：

      - **vocab_size** (`int`) –  嵌入字典的大小。
      - **embedding_size** (`int`) -  每个嵌入向量的大小。
      - **use_one_hot** (`bool`) –  指定是否使用one_hot编码形式。默认值：False。
      - **embedding_table** (`Union[Tensor, str, Initializer, numbers.Number]`) – embedding_table的初始化方法。指定字符串时，请参阅类初始化方法所对应的字符串值。默认值：'normal'。
      - **dtype** (`mindspore.dtype`) –  x的数据类型。默认值：mindspore.float32。
      - **padding_idx** (`int, None`) –  将 `padding_idx` 对应索引的输出嵌入向量用零填充。默认值：None。该功能已停用。

   **输入** ：

      - **x** （`tensor`) - 张量的形状  :math:`(\text{batch_size}, \text{x_length})` 。张量元素必须是整型值，并且元素数目必须小于等于vocab_size，否则相应的嵌入向量将为零。该数据类型可以是int32或int64。

   **输出** ：

      张量的形状  :math:`(\text{batch_size}, \text{x_length}, \text{embedding_size})` 。

   **异常** ：

      - TypeError – 如果 `vocab_size` 或者 `embedding_size` 不是整型值。

      - TypeError – 如果 `use_one_hot` 不是布尔值。

      - ValueError –  如果 `padding_idx` 是一个不在[0, `vocab_size` ]范围内的整数

   **支持平台** ：

      `Ascend` `GPU` `CPU`

   **样例** :

      .. code-block::

              >>> net = nn.Embedding(20000, 768,  True)
              >>> x = Tensor(np.ones([8, 128]), mindspore.int32)
              >>> # 将输入词的索引映射到词嵌入。
              >>> output = net(x)
              >>> result = output.shape
              >>> print(result)
              (8, 128, 768)