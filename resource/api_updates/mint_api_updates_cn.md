# mindspore.mint API接口变更

与上一版本相比，MindSpore中`mindspore.mint`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.mint.baddbmm](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.baddbmm.html#mindspore.mint.baddbmm)|New|对输入的两个三维矩阵batch1与batch2相乘，并将结果与input相加。|r2.4.0: Ascend|BLAS和LAPACK运算
[mindspore.mint.trace](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.trace.html#mindspore.mint.trace)|New|返回 input 的主对角线方向上的总和。|r2.4.0: Ascend|BLAS和LAPACK运算
[mindspore.mint.argmin](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.argmin.html#mindspore.mint.argmin)|New|返回输入Tensor在指定轴上的最小值索引。|r2.4.0: Ascend|Reduction运算
[mindspore.mint.median](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.median.html#mindspore.mint.median)|New|输出指定维度 dim 上的中值与其对应的索引。|r2.4.0: Ascend|Reduction运算
[mindspore.mint.distributed.destroy_process_group](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.distributed.destroy_process_group.html#mindspore.mint.distributed.destroy_process_group)|New|销毁指定通讯group。|r2.4.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.get_rank](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.distributed.get_rank.html#mindspore.mint.distributed.get_rank)|New|在指定通信组中获取当前的设备序号。|r2.4.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.get_world_size](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.distributed.get_world_size.html#mindspore.mint.distributed.get_world_size)|New|获取指定通信组实例的rank_size。|r2.4.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.init_process_group](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.distributed.init_process_group.html#mindspore.mint.distributed.init_process_group)|New|初始化通信服务并创建默认通讯group（group=GlobalComm.WORLD_COMM_GROUP）。|r2.4.0: Ascend|mindspore.mint.distributed
[mindspore.mint.cummax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.cummax.html#mindspore.mint.cummax)|New|返回一个元组（最值、索引），其中最值是输入Tensor input 沿维度 dim 的累积最大值，索引是每个最大值的索引位置。|r2.4.0: Ascend|其他运算
[mindspore.mint.cummin](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.cummin.html#mindspore.mint.cummin)|New|返回一个元组（最值、索引），其中最值是输入Tensor input 沿维度 dim 的累积最小值，索引是每个最小值的索引位置。|r2.4.0: Ascend|其他运算
[mindspore.mint.flatten](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.flatten.html#mindspore.mint.flatten)|New|沿着从 start_dim 到 end_dim 的维度，对输入Tensor进行展平。|r2.4.0: Ascend|其他运算
[mindspore.mint.tril](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.tril.html#mindspore.mint.tril)|New|返回输入Tensor input 的下三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。|r2.4.0: Ascend|其他运算
[mindspore.mint.full](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.full.html#mindspore.mint.full)|New|创建一个指定shape的Tensor，并用指定值填充。|r2.4.0: Ascend|创建运算
[mindspore.mint.linspace](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.linspace.html#mindspore.mint.linspace)|New|返回一个在区间 start 和 end （包括 start 和 end ）内均匀分布的，包含 steps 个值的一维Tensor。|r2.4.0: Ascend|创建运算
[mindspore.mint.nn.GroupNorm](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.GroupNorm.html#mindspore.mint.nn.GroupNorm)|New|在mini-batch输入上进行组归一化。|r2.4.0: Ascend|归一化层
[mindspore.mint.nn.L1Loss](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.L1Loss.html#mindspore.mint.nn.L1Loss)|New|L1Loss用于计算预测值和目标值之间的平均绝对误差。|r2.4.0: Ascend|损失函数
[mindspore.mint.nn.MSELoss](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.MSELoss.html#mindspore.mint.nn.MSELoss)|New|用于计算预测值与标签值之间的均方误差。|r2.4.0: Ascend|损失函数
[mindspore.mint.nn.functional.l1_loss](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.l1_loss.html#mindspore.mint.nn.functional.l1_loss)|New|用于计算预测值和目标值之间的平均绝对误差。|r2.4.0: Ascend|损失函数
[mindspore.mint.nn.functional.mse_loss](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.mse_loss.html#mindspore.mint.nn.functional.mse_loss)|New|计算预测值和标签值之间的均方误差。|r2.4.0: Ascend|损失函数
[mindspore.mint.nn.functional.avg_pool2d](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.avg_pool2d.html#mindspore.mint.nn.functional.avg_pool2d)|New|在输入Tensor上应用2D平均池化，输入Tensor可以看作是由一系列2D平面组成的。|r2.4.0: Ascend|池化函数
[mindspore.mint.nn.AvgPool2d](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.AvgPool2d.html#mindspore.mint.nn.AvgPool2d)|New|对输入张量应用二维平均池化，可视为二维输入平面的组合。|r2.4.0: Ascend|池化层
[mindspore.mint.masked_select](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.masked_select.html#mindspore.mint.masked_select)|New|返回一个一维Tensor，其中的内容是 input 中对应于 mask 中True位置的值。|r2.4.0: Ascend|索引、切分、连接、突变运算
[mindspore.mint.scatter](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.scatter.html#mindspore.mint.scatter)|New|根据指定索引将 src 中的值更新到 input 中返回输出。|r2.4.0: Ascend|索引、切分、连接、突变运算
[mindspore.mint.acos](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.acos.html#mindspore.mint.acos)|New|逐元素计算输入Tensor的反余弦。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.acosh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.acosh.html#mindspore.mint.acosh)|New|逐元素计算输入Tensor的反双曲余弦。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.arccos](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.arccos.html#mindspore.mint.arccos)|New|mindspore.mint.acos() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.arccosh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.arccosh.html#mindspore.mint.arccosh)|New|mindspore.mint.acosh() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.arcsin](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.arcsin.html#mindspore.mint.arcsin)|New|mindspore.mint.asin() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.arcsinh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.arcsinh.html#mindspore.mint.arcsinh)|New|mindspore.mint.asinh() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.arctan](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.arctan.html#mindspore.mint.arctan)|New|mindspore.mint.atan() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.arctanh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.arctanh.html#mindspore.mint.arctanh)|New|mindspore.mint.atanh() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.asin](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.asin.html#mindspore.mint.asin)|New|逐元素计算输入Tensor的反正弦。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.asinh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.asinh.html#mindspore.mint.asinh)|New|计算输入元素的反双曲正弦。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.atan](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.atan.html#mindspore.mint.atan)|New|逐元素计算输入Tensor的反正切值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.atanh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.atanh.html#mindspore.mint.atanh)|New|逐元素计算输入Tensor的反双曲正切值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.bitwise_and](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.bitwise_and.html#mindspore.mint.bitwise_and)|New|逐元素执行两个Tensor的与运算。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.bitwise_or](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.bitwise_or.html#mindspore.mint.bitwise_or)|New|逐元素执行两个Tensor的或运算。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.bitwise_xor](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.bitwise_xor.html#mindspore.mint.bitwise_xor)|New|逐元素执行两个Tensor的异或运算。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.cosh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.cosh.html#mindspore.mint.cosh)|New|逐元素计算 input 的双曲余弦值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.cross](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.cross.html#mindspore.mint.cross)|New|返回沿着维度 dim 上，input 和 other 的向量积（叉积）。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.erfc](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.erfc.html#mindspore.mint.erfc)|New|逐元素计算 input 的互补误差函数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.expm1](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.expm1.html#mindspore.mint.expm1)|New|逐元素计算输入Tensor的指数，然后减去1。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.fix](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.fix.html#mindspore.mint.fix)|New|mindspore.mint.trunc() 的别名。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.log1p](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.log1p.html#mindspore.mint.log1p)|New|对输入Tensor逐元素加一后计算自然对数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.logical_xor](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.logical_xor.html#mindspore.mint.logical_xor)|New|逐元素计算两个Tensor的逻辑异或运算。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.remainder](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.remainder.html#mindspore.mint.remainder)|New|逐元素计算 input 除以 other 后的余数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.roll](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.roll.html#mindspore.mint.roll)|New|沿轴移动Tensor的元素。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.round](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.round.html#mindspore.mint.round)|New|对输入数据进行四舍五入到最接近的整数数值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.sign](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.sign.html#mindspore.mint.sign)|New|按sign公式逐元素计算输入Tensor。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.sinc](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.sinc.html#mindspore.mint.sinc)|New|计算输入的归一化正弦值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.sinh](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.sinh.html#mindspore.mint.sinh)|New|逐元素计算输入Tensor的双曲正弦。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.special.erfc](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.special.erfc.html#mindspore.mint.special.erfc)|New|逐元素计算 input 的互补误差函数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.special.expm1](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.special.expm1.html#mindspore.mint.special.expm1)|New|逐元素计算输入Tensor的指数，然后减去1。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.special.log1p](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.special.log1p.html#mindspore.mint.special.log1p)|New|对输入Tensor逐元素加一后计算自然对数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.special.log_softmax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.special.log_softmax.html#mindspore.mint.special.log_softmax)|New|在指定轴上对输入Tensor应用LogSoftmax函数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.special.round](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.special.round.html#mindspore.mint.special.round)|New|对输入数据进行四舍五入到最接近的整数数值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.special.sinc](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.special.sinc.html#mindspore.mint.special.sinc)|New|计算输入的归一化正弦值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.tan](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.tan.html#mindspore.mint.tan)|New|逐元素计算输入元素的正切值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.trunc](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.trunc.html#mindspore.mint.trunc)|New|返回一个新的Tensor，该Tensor具有输入元素的截断整数值。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.xlogy](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.xlogy.html#mindspore.mint.xlogy)|New|计算第一个输入乘以第二个输入的对数。|r2.4.0: Ascend|逐元素运算
[mindspore.mint.nn.functional.hardshrink](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.hardshrink.html#mindspore.mint.nn.functional.hardshrink)|New|Hard Shrink激活函数。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.hardsigmoid](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.hardsigmoid.html#mindspore.mint.nn.functional.hardsigmoid)|New|Hard Sigmoid激活函数。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.hardswish](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.hardswish.html#mindspore.mint.nn.functional.hardswish)|New|Hard Swish激活函数。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.log_softmax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.log_softmax.html#mindspore.mint.nn.functional.log_softmax)|New|在指定轴上对输入Tensor应用LogSoftmax函数。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.mish](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.mish.html#mindspore.mint.nn.functional.mish)|New|逐元素计算输入Tensor的MISH（A Self Regularized Non-Monotonic Neural Activation Function 自正则化非单调神经激活函数）。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.prelu](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.prelu.html#mindspore.mint.nn.functional.prelu)|New|带参数的线性修正单元激活函数（Parametric Rectified Linear Unit activation function）。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.selu](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.selu.html#mindspore.mint.nn.functional.selu)|New|激活函数selu（Scaled exponential Linear Unit）。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.softshrink](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.functional.softshrink.html#mindspore.mint.nn.functional.softshrink)|New|Soft Shrink激活函数。|r2.4.0: Ascend|非线性激活函数
[mindspore.mint.nn.GELU](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.GELU.html#mindspore.mint.nn.GELU)|New|高斯误差线性单元激活函数（Gaussian Error Linear Unit）。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Hardshrink](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.Hardshrink.html#mindspore.mint.nn.Hardshrink)|New|逐元素计算Hard Shrink激活函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Hardsigmoid](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.Hardsigmoid.html#mindspore.mint.nn.Hardsigmoid)|New|逐元素计算Hard Sigmoid激活函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Hardswish](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.Hardswish.html#mindspore.mint.nn.Hardswish)|New|逐元素计算Hard Swish激活函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.LogSoftmax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.LogSoftmax.html#mindspore.mint.nn.LogSoftmax)|New|在指定轴上对输入Tensor应用LogSoftmax函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Mish](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.Mish.html#mindspore.mint.nn.Mish)|New|逐元素计算输入Tensor的MISH（A Self Regularized Non-Monotonic Neural Activation Function 自正则化非单调神经激活函数）。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.PReLU](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.PReLU.html#mindspore.mint.nn.PReLU)|New|逐元素计算PReLU（PReLU Activation Operator）激活函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.ReLU](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.ReLU.html#mindspore.mint.nn.ReLU)|New|逐元素计算ReLU（Rectified Linear Unit activation function）修正线性单元激活函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.SELU](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.SELU.html#mindspore.mint.nn.SELU)|New|逐元素计算激活函数SELU（Scaled exponential Linear Unit）。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Softmax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.Softmax.html#mindspore.mint.nn.Softmax)|New|将Softmax函数应用于n维输入Tensor。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Softshrink](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/mint/mindspore.mint.nn.Softshrink.html#mindspore.mint.nn.Softshrink)|New|逐元素计算Soft Shrink激活函数。|r2.4.0: Ascend|非线性激活层 (加权和，非线性)
