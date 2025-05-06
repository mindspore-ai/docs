# mindspore.mint API接口变更

与上一版本2.5.0相比，MindSpore中`mindspore.mint`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.mint.nn.functional.pixel_shuffle](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.pixel_shuffle.html#mindspore.mint.nn.functional.pixel_shuffle)|New|根据上采样系数重排Tensor中的元素。|r2.6.0rc1: Ascend|Vision函数
[mindspore.mint.nn.PixelShuffle](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.PixelShuffle.html#mindspore.mint.nn.PixelShuffle)|New|根据上采样系数重新排列Tensor中的元素。|r2.6.0rc1: Ascend|Vision层
[mindspore.mint.distributed.is_available](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.distributed.is_available.html#mindspore.mint.distributed.is_available)|New|分布式模块是否可用。|r2.6.0rc1: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.is_initialized](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.distributed.is_initialized.html#mindspore.mint.distributed.is_initialized)|New|默认的通信组是否初始化。|r2.6.0rc1: Ascend|mindspore.mint.distributed
[mindspore.mint.optim.SGD](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.optim.SGD.html#mindspore.mint.optim.SGD)|New|随机梯度下降算法。|r2.6.0rc1: Ascend|mindspore.mint.optim
[mindspore.mint.diag](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.diag.html#mindspore.mint.diag)|New|如果 input 是向量（1-D 张量），则返回一个二维张量，其中 input 的元素作为对角线。|r2.6.0rc1: Ascend|其他运算
[mindspore.mint.triangular_solve](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.triangular_solve.html#mindspore.mint.triangular_solve)|New|求解正上三角形或下三角形可逆矩阵 A 和包含多个元素的右侧边 b 的方程组的解。|r2.6.0rc1: Ascend|其他运算
[mindspore.mint.nn.KLDivLoss](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.KLDivLoss.html#mindspore.mint.nn.KLDivLoss)|New|计算输入 input 和 target 的Kullback-Leibler散度。|r2.6.0rc1: Ascend|损失函数
[mindspore.mint.nn.functional.cross_entropy](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.cross_entropy.html#mindspore.mint.nn.functional.cross_entropy)|New|获取预测值和目标值之间的交叉熵损失。|r2.6.0rc1: Ascend|损失函数
[mindspore.mint.nn.functional.kl_div](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.kl_div.html#mindspore.mint.nn.functional.kl_div)|New|计算输入 input 和 target 的Kullback-Leibler散度。|r2.6.0rc1: Ascend|损失函数
[mindspore.mint.nn.functional.adaptive_avg_pool3d](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.adaptive_avg_pool3d.html#mindspore.mint.nn.functional.adaptive_avg_pool3d)|New|对一个多平面输入信号执行三维自适应平均池化。|r2.6.0rc1: Ascend|池化函数
[mindspore.mint.nn.functional.adaptive_max_pool1d](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.adaptive_max_pool1d.html#mindspore.mint.nn.functional.adaptive_max_pool1d)|New|对一个多平面输入信号执行一维自适应最大池化。|r2.6.0rc1: Ascend|池化函数
[mindspore.mint.nn.functional.avg_pool3d](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.avg_pool3d.html#mindspore.mint.nn.functional.avg_pool3d)|New|在输入Tensor上应用3d平均池化，输入Tensor可以看作是由一系列3d平面组成的。|r2.6.0rc1: Ascend|池化函数
[mindspore.mint.nn.AdaptiveMaxPool1d](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.AdaptiveMaxPool1d.html#mindspore.mint.nn.AdaptiveMaxPool1d)|New|对由多个输入平面组成的输入信号应用1D自适应最大池化。|r2.6.0rc1: Ascend|池化层
[mindspore.mint.nn.AvgPool3d](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.AvgPool3d.html#mindspore.mint.nn.AvgPool3d)|New|对输入张量应用三维平均池化，可视为三维输入平面的组合。|r2.6.0rc1: Ascend|池化层
[mindspore.mint.index_add](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.index_add.html#mindspore.mint.index_add)|New|根据 index 中的索引顺序，将 alpha 乘以 source 的元素累加到 input 中。|r2.6.0rc1: Ascend|索引、切分、连接、突变运算
[mindspore.mint.linalg.qr](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.linalg.qr.html#mindspore.mint.linalg.qr)|New|对输入矩阵进行正交分解： $(A = QR)$。|r2.6.0rc1: Ascend|逆数
[mindspore.mint.logaddexp2](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.logaddexp2.html#mindspore.mint.logaddexp2)|New|计算以2为底的输入的指数和的对数。|r2.6.0rc1: Ascend|逐元素运算
[mindspore.mint.nn.functional.elu_](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.elu_.html#mindspore.mint.nn.functional.elu_)|New|指数线性单元激活函数。|r2.6.0rc1: Ascend|非线性激活函数
[mindspore.mint.nn.functional.glu](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.functional.glu.html#mindspore.mint.nn.functional.glu)|New|计算输入Tensor的门线性单元激活函数（Gated Linear Unit activation function）值。|r2.6.0rc1: Ascend|非线性激活函数
[mindspore.mint.nn.GLU](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.GLU.html#mindspore.mint.nn.GLU)|New|计算输入Tensor的门线性单元激活函数（Gated Linear Unit activation function）值。|r2.6.0rc1: Ascend|非线性激活层 (加权和，非线性)
[mindspore.mint.nn.Sigmoid](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mint/mindspore.mint.nn.Sigmoid.html#mindspore.mint.nn.Sigmoid)|New|逐元素计算Sigmoid激活函数。|r2.6.0rc1: Ascend|非线性激活层 (加权和，非线性)
