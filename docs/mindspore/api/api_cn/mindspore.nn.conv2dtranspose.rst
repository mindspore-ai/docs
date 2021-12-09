mindspore.nn.Conv2dTranspose
============================

.. py:class:: mindspore.nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros")

   二维转置卷积层。

   计算一个二维转置卷积，这也被称为反卷积（实际不是真正的反卷积）。

   `x` 的shape通常为 :math:`(N, C, H, W)` ，其中 :math:`N` 是batch大小，:math:`C` 是通道数。如果'pad_mode'设为 "pad"，输出的高度和宽度分别为：

   .. math:: \begin{align}\begin{aligned}H_{out} = (H_{in} - 1) \times \text{stride[0]} - \left (\text{padding[0]} + \text{padding[1]}\right ) + \text{dilation[0]} \times (\text{kernel_size[0]} - 1) + 1\\W_{out} = (W_{in} - 1) \times \text{stride[1]} - \left (\text{padding[2]} + \text{padding[3]}\right ) + \text{dilation[1]} \times (\text{kernel_size[1]} - 1) + 1\end{aligned}\end{align}

   其中 :math:`\text{kernel_size[0]}` 是卷积核的高度， :math:`\text{kernel_size[1]}` 是卷积核的宽度。

   **参数** ：

      - **in_channels** (`int`) – 输入空间的通道数。
      - **out_channels** (`dict`) - 输出空间的通道数。
      - **kernel_size** (`Union[int, tuple[int]]`) – 该参数指定二维卷积核的高度和宽度。数据类型为整型或2个整型的元组。一个整数表示卷积核的高度和宽度均为该值。2个整数的元组分别表示卷积核高度和宽度。
      - **stride** (`Union[int, tuple[int]]`) – 步长大小。数据类型为整型或2个整型的元组。一个整数表示在高度和宽度方向的滑动步长均为该值。2个整数的元组分别表示在高度和宽度方向的滑动步长。默认值：1。

      - **pad_mode** (`str`) –

         指定填充模式。可选值为“pad”，“same”，“valid”。默认值：“same”。

         - **pad** ：输入 `x` 两侧的隐式填充。
         - **same** ：采用补全方式。
         - **valid** ：采用丢弃方式。

      - **padding** (`Union[int, tuple[int]]`) – 输入 `x` 两侧的隐式填充。 如果 `padding` 是一个整数，那么上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有四个整数的元组，那么上、下、左、右的填充分别等于 `padding[0]`、`padding[1]`、`padding[2]` 和 `padding[3]`。默认值：0。
      - **dilation** (`Union[int, tuple[int]]`) – 指定用于膨胀卷积的膨胀率。数据类型为整型或具有2个整型的元组。如果设置 :math:`k > 1`，则每个采样位置将跳过 :math:`k-1` 个像素。其值必须大于或等于1，并以输入的高度和宽度为边界。默认值：1。
      - **group** (`int`) –  将过滤器分组， `in_channels` 和 `out_channels` 必须被组数整除。当组数大于1时，不支持达芬奇（Davinci）设备。默认值：1.
      - **has_bias** (`bool`) –  指定图层是否使用偏置向量。默认值：False。
      - **weight_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 卷积核的初始化方法。它可以是张量，字符串，初始化实例或数字。当使用字符串时，可选“TruncatedNormal”，“Normal”，“Uniform”，“HeUniform”和“XavierUniform”分布以及常量“One”和“Zero”分布的值，可接受别名“ xavier_uniform”，“ he_uniform”，“ ones”和“ zeros”。上述字符串大小写均可。更多细节请参考Initializer的值。默认值：“normal”。
      - **bias_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 偏置向量的初始化方法。可以使用的初始化方法和字符串与“weight_init”相同。更多细节请参考Initializer的值。默认值：“zeros”。

   **输入** ：

      - **x** (Tensor) - Shape 为 :math:`(N, C_{in}, H_{in}, W_{in})` 的张量。

   **输出** ：

      Shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 的张量。

   **异常** ：

      - **TypeError** - 如果 `in_channels`，`out_channels` 或者 `group` 不是整数。
      - **TypeError** - 如果 `kernel_size`，`stride`，`padding` 或者 `dilation` 既不是整数也不是元组。
      - **ValueError** - 如果 `in_channels`，`out_channels`，`kernel_size`，`stride` 或者 `dilation` 小于1。
      - **ValueError** - 如果 `padding` 小于0。
      - **ValueError** - 如果 `pad_mode` 不是’same’，‘valid’，‘pad’中的一个。
      - **ValueError** - 如果 `padding` 是一个长度不等于4的元组。
      - **ValueError** - 如果 `pad_mode` 不等于‘pad’且 `padding` 不等于(0,0,0,0)。

   **支持平台** ：

      `Ascend` `GPU` `CPU`

   **样例** :

      .. code-block::

              >>> net = nn.Conv2dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
              >>> x = Tensor(np.ones([1, 3, 16, 50]), mindspore.float32)
              >>> output = net(x).shape
              >>> print(output)
              (1, 64, 19, 53)