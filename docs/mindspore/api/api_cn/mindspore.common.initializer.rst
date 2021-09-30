mindspore.common.initializer
=============================
初始化神经元参数

.. py:class:: mindspore.common.initializer.Initializer(**kwargs)
   
    初始化器的基类，表示用于初始化张量的数据类型。
   
    **参数：**

        - **kwargs** (`dict`) – **Initializer**的关键字参数。

    **返回：**

        Array，初始化后的数组。
   
.. py:method:: mindspore.common.initializer.initializer(init, shape=None, dtype=mstype.float32)
   
    创建并初始化一个张量。
   
    **参数：**

        - **init** (`Union[Tensor, str, Initializer子类, numbers.Number]`) – 初始化方式。

            - *str*：`init`是继承自 Initializer 的类的别名，相应的类将被调用。 `init`的值可以是“normal”、“ones”或“zeros”等。

            - *Initializer*：`init`是从 Initializer 继承来初始化张量的类。
            
            - *numbers.Number*：调用常量来初始化张量。
            
        - **shape** (`Union[[tuple, list, int]`) - 初始化后的形状，可传入整数类型的列表、元组和变量作为参数，默认值为None。
        
        - **dtype** (`mindspore.dtype`) – 初始化后张量内的数据类型，默认值为`mindspore.float32`。 

    **返回：**

        Union[Tensor]，返回一个张量对象。
        
    **样例：**
    
    .. code-block::
    
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, One        
        >>> tensor = initializer('ones', [1, 2, 3], mindspore.float32)       
        >>> tensor = initializer(One(), [1, 2, 3], mindspore.float32)       
        >>> tensor = initializer(0, [1, 2, 3], mindspore.float32)
        
.. py:class:: mindspore.common.initializer.TruncatedNormal(sigma=0.01)

    初始化一个截断的正态分布数组，具有固定的上下界，记为N(low, high)。
    
    **参数：**

        - **sigma** (`float`) - 正态分布数组的标准差，默认值为0.01。
        
    **返回：**

        Array，截断的正态分布数组。
        
.. py:class:: mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)

    初始化一个正态分布数组，使用均数和标准差来确定张量内填充的数值，记为N(sigma, mean)。

    .. math::
    f(x) =  \frac{1} {\sqrt{2*π} * sigma}exp(-\frac{(x - mean)^2} {2*{sigma}^2})
     
    **参数：**

        - **sigma** (`float`) - 正态分布数组的标准差，默认值为0.01。

        - **mean** (`float`) - 正态分布数组的均数，默认值为0.0。
    
    **返回：**

        Array，正态分布数组。
        
.. py:class:: mindspore.common.initializer.Uniform(scale=0.07)

    初始化一个均匀分布数组，使用对称的上下界（scale）来确定张量内填充的数值，记为U(-scale, scale)。
    
    **参数：**

        - **scale** (`float`) - 均匀分布数组的边界，默认值为0.07。
    
    **返回：**

        Array，均匀分布数组。

.. py:class:: mindspore.common.initializer.HeUniform(negative_slope=0, mode="fan_in", nonlinearity="leaky_relu")

    用HeUniform方法初始化一个数组，数组内的样本符合均匀分布U[-boundary,boundary]。
	
    边界（boundary）的定义： 
	
    .. math::
        boundary = \sqrt{\frac{6}{(1 + a^2) \times \text{fan_in}}}
    
    **参数：**

        - **negative_slope** (`int, float, bool`) - 本层后激活函数的负数区间斜率（仅适用于非线性激活函数‘leaky_relu’），默认值为0。

        - **mode** (`str`) - 可选“fan_in”或“fan_out”，“fan_in”会保留前向传递中权重的方差大小，“fan_out”会保留反向传递的数值，默认为”fan_in“。
        
        - **nonlinearity** (`str`) - 非线性函数，仅有“relu”或“leaky_relu”可供选择，默认为“leaky_relu”。
        
    **返回：**

        Array，HeUniform数据。
        
.. py:class:: mindspore.common.initializer.HeNormal(negative_slope=0, mode="fan_in", nonlinearity="leaky_relu")

    用HeNormal方法初始化一个数组，数组内的样本符合正态分布N(0, sigma)。

    .. math::
        sigma = \frac{gain} {\sqrt{mode}}
    
    其中，
    
    gain是一个可选的缩放因子。mode 是权重张量中输入单元或输出单元的数量。

    HeUniform 算法的详细信息，请查看 https://arxiv.org/abs/1502.01852。
    
    **参数：**

        - **negative_slope** (`int, float, bool`) - 本层后激活函数的负数区间斜率（仅适用于非线性激活函数‘leaky_relu’），默认值为0。

        - **mode** (`str`) - 可选“fan_in”或“fan_out”，“fan_in”会保留前向传递中权重的方差大小，“fan_out”会保留向后传递的数值，默认为”fan_in“。
        
        - **nonlinearity** (`str`) - 非线性函数，仅有“relu”或“leaky_relu”可供选择，默认为“leaky_relu”。
        
    **返回：**

        Array，HeNormal数据。
        
.. py:class:: mindspore.common.initializer.XavierUniform(gain=1)

    用Xarvier方法分布初始化一个数组，样本符合均匀分布U[-boundary,boundary]。
	
    边界（boundary）的定义如下：
    
    .. math::

    boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}
	
    - gain是一个可选的缩放因子。
    - n_{in}为权重张量内输入单元的数量。
    - n_{out}为权重张量内输出单元的数量。

    有关 XavierUniform 算法的详细信息，请查看 http://proceedings.mlr.press/v9/glorot10a.html。
    
     **参数：** 

        - **gain** (`float`) - 可选的缩放因子，默认值为1。
    
     **返回：**

        Array，Xarvier均匀分布数组。
        
.. py:class:: mindspore.common.initializer.One(**kwargs)

    初始化一个全为1的数组。
    
    **参数：**

        - **arr** (`Array`) - 未初始化的数组。
    
    **返回：**

        Array，全为1的数组。
    
.. py:class:: mindspore.common.initializer.Zero(**kwargs)

    初始化一个全为0的数组。
    
    **参数：**

        - **arr** (`Array`) - 未初始化的数组。
    
    **返回：**

        Array，全为0的数组。
        
.. py:class:: mindspore.common.initializer.Constant(value)
    
    初始化一个常数数组。
    
    **参数：**

        - **value** (`Union[int, numpy.ndarray]`) - 用于初始化的常数值或者ndarry数组。
    
    **返回：**

        Array，指定常数的数组。