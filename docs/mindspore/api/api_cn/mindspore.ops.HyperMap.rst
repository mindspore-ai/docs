mindspore.ops.HyperMap
=======================

.. py:class:: mindspore.ops.HyperMap(ops=None, reverse=False)

   *HyperMap* 可以对输入序列做集合运算。
   
   将运算应用于序列或嵌套序列中的每个元素。与 *Map* 不同，*HyperMap* 能够用于嵌套结构。

   **参数** ：

      - **ops** (Union[`MultitypeFuncGraph`, `None`]) –  *ops* 是指定的运算操作。如果 *ops* 为 `None` ，则运算应该作为 *HyperMap* 实例的第一个入参。默认值为 `None` 。
      - **reverse** (`bool`) -  在某些场景下，优化器需要倒置以提高并行性能，一般情况下，用户可以忽略。*reverse* 用于决定是否逆向执行运算，仅在图模式下支持。默认值为False。

   **输入** ：

      - **args** (Tuple[sequence]) -  如果 *ops* 不是 `None` ，则所有入参都应该是具有相同长度的序列，并且序列的每一行都是运算的输入。
        如果 *ops* 是 `None` ，则第一个入参是运算，其余都是输入。

   **输出** ：

      序列或嵌套序列，执行函数如 `operation(args[0][i], args[1][i])` 之后输出的序列。

   **异常** ：

      **TypeError** - 如果 *ops* 不是 `MultitypeFuncGraph` 或 `None` 。

      **TypeError** - 如果 *args* 不是一个 `Tuple` 。

   **支持平台** ：

      `Ascend` `GPU` `CPU`

   **样例** :

      .. code-block::

              >>> from mindspore import dtype as mstype
              >>> nest_tensor_list = ((Tensor(1, mstype.float32), Tensor(2, mstype.float32)),
              ...                     (Tensor(3, mstype.float32), Tensor(4, mstype.float32)))
              >>> # 对嵌套列表中的所有Tensor进行平方运算
              >>>
              >>> square = MultitypeFuncGraph('square')
              >>> @square.register("Tensor")
              ... def square_tensor(x):
              ...     return ops.square(x)
              >>>
              >>> common_map = HyperMap()
              >>> output = common_map(square, nest_tensor_list)
              >>> print(output)
              ((Tensor(shape=[], dtype=Float32, value= 1), Tensor(shape=[], dtype=Float32, value= 4)),
              (Tensor(shape=[], dtype=Float32, value= 9), Tensor(shape=[], dtype=Float32, value= 16)))
              >>> square_map = HyperMap(square, False)
              >>> output = square_map(nest_tensor_list)
              >>> print(output)
              ((Tensor(shape=[], dtype=Float32, value= 1), Tensor(shape=[], dtype=Float32, value= 4)),
              (Tensor(shape=[], dtype=Float32, value= 9), Tensor(shape=[], dtype=Float32, value= 16)))
