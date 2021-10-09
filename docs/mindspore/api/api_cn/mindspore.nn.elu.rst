mindspore.nn.ELU
=================

.. py:class:: mindspore.nn.ELU(alpha=1.0)

   指数线性单元激活函数。

   ELU函数应用于输入的每个元素。该激活函数定义如下：
   
   .. math::
         E_{i} =
         \begin{cases}
         x, &\text{if } x \geq 0; \cr
         \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
         \end{cases}


   关于ELU图像可参考 `ELU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_elu.svg>`_  。

   **参数** ：

      - **alpha** (`float`) – 负因子系数，数据类型为浮点数。默认值：1.0。

   **输入** ：

      - **x** （Tensor） - ELU的输入数据类型为float16或float32。形状是 :math:`(N,*)` ，:math:`*` 表示任意数量的附加维度。

   **输出** ：

      张量，跟x保持相同的数据类型和形状。

   **异常** ：

   - TypeError – 如果 `alpha` 不是浮点数。

   - TypeError – 如果 `x` 的数据类型既不是float16也不是float32。

   - ValueError –  如果 `alpha` 不等于1.0。

   **支持平台** ：

      `Ascend` `GPU` `CPU`

   **样例** :

      .. code-block::

              >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
              >>> elu = nn.ELU()
              >>> result = elu(x)
              >>> print(result)
              [-0.63212055  -0.86466473  0.  2.  1.]