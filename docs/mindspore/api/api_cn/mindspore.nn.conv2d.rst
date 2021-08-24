mindspore.nn.Conv2d
====================

.. py:class:: mindspore.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")

   二维卷积层。

   对输入张量进行二维卷积，该张量的常见shape是 :math:`(N, C_{in}, H_{in}, W_{in})`，其中 :math:`N` 为batch大小，:math:`C_{in}` 为通道数，:math:`H_{in},W_{in}` 为高度和宽度。对于每个batch中shape为 :math:`(C_{in}, H_{in}, W_{in})` 的张量输入，公式定义如下：

   .. math:: out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

   其中 :math:`corr` 是互关联算子，:math:`C_{in}` 是输入通道数目，:math:`j` 的范围在 :math:`[0，C_{out}-1]` 内，:math:`W_{ij}`对应第 :math:`j`个的过滤器的第 :math:`i` 个通道，:math:`out_j`对应输出的第 :math:`j` 个通道。:math:`W_{ij}` 是shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})` 的kernel切片。其中 :math:`\text{kernel_size[0]}` 和 :math:`\text{kernel_size[1]}` 是卷积kernel的高度和宽度。完整kernel的shape是 :math:`(C_{out}, C_{in} // \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`，其中group是在通道维度上分割输入 `x` 的组数。
   如果'pad_mode'被设置为 "valid"，输出高度和宽度分别为 :math:`\left \lfloor{1 + \frac{H_{in} + \text{padding[0]} + \text{padding[1]} - \text{kernel_size[0]} -
   (\text{kernel_size[0]} - 1) \times (\text{dilation[0]} - 1) }{\text{stride[0]}}} \right \rfloor` 和 :math:`\left \lfloor{1 + \frac{W_{in} + \text{padding[2]} + \text{padding[3]} - \text{kernel_size[1]} -
   (\text{kernel_size[1]} - 1) \times (\text{dilation[1]} - 1) }{\text{stride[1]}}} \right \rfloor`。

   详细介绍请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

   **参数** ：

      - **in_channels** (`int`) – 输入的通道数 :math:`C_{in}`。
      - **out_channels** (`dict`) - 输出的通道数 :math:`C_{out}`。
      - **kernel_size** (`Union[int, tuple[int]]`) – 该参数指定二维卷积核的高度和宽度。数据类型为整型值或2个整型值的元组。一个整数表示卷积核的高度和宽度均为该值。2个整数的元组分别表示卷积核高度和宽度。
      - **stride** (`Union[int, tuple[int]]`) – 步长大小。数据类型为整型值或2个整型值的元组。一个整数表示在高度和宽度方向的滑动步长均为该值。2个整数的元组分别表示在高度和宽度方向的滑动步长。默认值：1。

      - **pad_mode** (`str`) –

         指定填充模式。可选值为“same”，“valid”，“pad”。默认值：“same”。

         - **same** ：采用补全方式。输出的高度与宽度与输入 `x` 一致。将计算水平和垂直方向填充总数。并在可能的情况下均匀分布到顶部、底部、左侧和右侧。否则最后额外填充将从底部到右侧开始。若设置该模式，`padding` 必须为0。
         - **valid** ：采用丢弃方式。在不填充前提下返回可能的最大高度和宽度的输出。多余的像素会被丢弃。若设置该模式，`padding` 必须为0。
         - **pad** ：输入 `x` 两侧的隐式填充，填充数量将填充到输入张量边界。`padding` 应大于或等于0。

      - **padding** (`Union[int, tuple[int]]`) –  输入 `x` 两侧的隐式填充。 如果 `padding` 是一个整数，那么上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有四个整数的元组，那么上、下、左、右的填充分别等于 `padding[0]`、`padding[1]`、`padding[2]` 和 `padding[3]`。默认值：0。
      - **dilation** (`Union[int, tuple[int]]`) –  指定用于膨胀卷积的膨胀率。数据类型为整型或具有2个整型值的元组。如果设置 :math:`k> 1`，则每个采样位置将跳过 :math:`k-1` 个像素。其值必须大于或等于1，并以输入的高度和宽度为边界。默认值：1。
      - **group** (`int`) –  将过滤器分组， `in_channels` 和 `out_channels` 必须被组数整除。如果组数等于 `in_channels` 和 `out_channels` ,这个二维卷积层也被称为二维深度卷积层。默认值：1.
      - **has_bias** (`bool`) –  指定图层是否使用偏置向量。默认值：False。
      - **weight_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 卷积核的初始化方法。它可以是张量，字符串，初始化实例或数字。当使用字符串时，可选“TruncatedNormal”，“Normal”，“Uniform”，“HeUniform”和“XavierUniform”分布以及常量“One”和“Zero”分布的值，可接受别名“ xavier_uniform”，“ he_uniform”，“ ones”和“ zeros”。上述字符串大小写均可。更多细节请参考Initializer的值。默认值：“normal”。
      - **bias_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 偏置向量的初始化方法。可以使用的初始化方法和字符串与“weight_init”相同。更多细节请参考Initializer的值。默认值：“zeros”。
      - **data_format** (`str`) –  数据格式的可选值有‘NHWC’，‘NCHW’。默认值：‘NCHW’。

   **输入** ：

      - **x** (Tensor) - Shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(N, H_{in}, W_{in}, C_{in})` 的张量。

   **输出** ：

      Shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或者 :math:`(N, H_{out}, W_{out}, C_{out})` 的张量。

   **抛出异常** ：

      - **TypeError** - 如果 `in_channels`，`out_channels` 或者 `group` 不是整型值。
      - **TypeError** - 如果 `kernel_size`，`stride`，`padding` 或者 `dilation` 既不是整型值也不是元组。
      - **ValueError** - 如果 `in_channels`，`out_channels`，`kernel_size`，`stride` 或者 `dilation` 小于1。
      - **ValueError** - 如果 `padding` 小于0。
      - **ValueError** - 如果 `pad_mode` 不是’same’，‘valid’，‘pad’其中之一。
      - **ValueError** - 如果 `padding` 是一个长度不等于4的元组。
      - **ValueError** - 如果 `pad_mode` 不等于‘pad’且 `padding` 不等于(0,0,0,0)。
      - **ValueError** - 如果 `data_format` 既不是‘NCHW’也不是‘NHWC’。


   **支持平台** ：

      `Ascend` `GPU` `CPU`

   **样例** :

      .. code-block::

              >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
              >>> x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
              >>> output = net(x).shape
              >>> print(output)
              (1, 240, 1024, 640)