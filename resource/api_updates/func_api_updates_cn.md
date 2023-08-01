# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
|[mindspore.ops.deepcopy](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.deepcopy.html#mindspore.ops.deepcopy)|New|返回输入Tensor的深拷贝。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.diagonal_scatter](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.diagonal_scatter.html#mindspore.ops.diagonal_scatter)|New|dim1 和 dim2 指定 input 的两个维度，这两个维度上的元素将被视为矩阵的元素，并且将 src 嵌入到该矩阵的对角线上。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.is_nonzero](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.is_nonzero.html#mindspore.ops.is_nonzero)|New|判断输入Tensor是否包含0或False，输入只能是单元素。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.nanmean](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.nanmean.html#mindspore.ops.nanmean)|New|计算 input 指定维度元素的平均值，忽略NaN。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.nanmedian](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.nanmedian.html#mindspore.ops.nanmedian)|New|计算 input 指定维度元素的中值和索引，忽略NaN。|r2.1: CPU|Array操作
|[mindspore.ops.roll](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.roll.html#mindspore.ops.roll)|New|沿轴移动Tensor的元素。|r2.1: GPU|Array操作
|[mindspore.ops.row_stack](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.row_stack.html#mindspore.ops.row_stack)|New|mindspore.ops.vstack() 的别名。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.select_scatter](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.select_scatter.html#mindspore.ops.select_scatter)|New|将 src 中的值散布到 input 指定维度 axis 的指定位置 index 上。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.slice_scatter](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.slice_scatter.html#mindspore.ops.slice_scatter)|New|指定维度上对输入Tensor进行切片并将源Tensor覆盖切片结果。|r2.1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.logcumsumexp](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.logcumsumexp.html#mindspore.ops.logcumsumexp)|New|计算输入Tensor input 元素的的指数沿轴 axis 的累积和的对数。|r2.1: Ascend/CPU/GPU|Reduction函数
|[mindspore.ops.soft_margin_loss](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.soft_margin_loss.html#mindspore.ops.soft_margin_loss)|New|计算 input 和 target 之间的soft margin loss。|r2.1: Ascend/GPU|损失函数
|[mindspore.ops.bucketize](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.bucketize.html#mindspore.ops.bucketize)|New|根据 boundaries 对 input 进行分桶。|r2.1: Ascend/GPU/CPU|比较函数
|[mindspore.ops.bidense](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.bidense.html#mindspore.ops.bidense)|New|对输入 input1 和 input2 应用双线性全连接操作。|r2.1: Ascend/GPU/CPU|神经网络
|[mindspore.ops.conv1d](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.conv1d.html#mindspore.ops.conv1d)|Changed|对输入Tensor计算一维卷积。|r2.0: Ascend/GPU/CPU => r2.1: Ascend/GPU|神经网络
|[mindspore.ops.conv2d](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.conv2d.html#mindspore.ops.conv2d)|Changed|对输入Tensor计算二维卷积。|r2.0: Ascend/GPU/CPU => r2.1: Ascend/GPU|神经网络
|[mindspore.ops.conv3d](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.conv3d.html#mindspore.ops.conv3d)|Changed|对输入Tensor计算三维卷积。|r2.0: Ascend/GPU/CPU => r2.1: Ascend/GPU|神经网络
|[mindspore.ops.dense](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.dense.html#mindspore.ops.dense)|New|对输入 input 应用全连接操作。|r2.1: Ascend/GPU/CPU|神经网络
|[mindspore.ops.cond](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.cond.html#mindspore.ops.cond)|New|返回给定Tensor的矩阵范数或向量范数。|r2.1: GPU/CPU|线性代数函数
|[mindspore.ops.eigvals](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.eigvals.html#mindspore.ops.eigvals)|New|计算输入方阵（batch方阵）的特征值。|r2.1: Ascend/CPU|线性代数函数
|[mindspore.ops.lu_solve](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.lu_solve.html#mindspore.ops.lu_solve)|New|给定LU分解结果 $A$ 和列向量 $b$，求解线性方程组的解y $Ay = b$。|r2.1: Ascend/GPU/CPU|线性代数函数
|[mindspore.ops.vander](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.vander.html#mindspore.ops.vander)|New|生成一个范德蒙矩阵。|r2.1: Ascend/GPU/CPU|线性代数函数
|[mindspore.ops.vecdot](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.vecdot.html#mindspore.ops.vecdot)|New|在指定维度上，计算两批向量的点积。|r2.1: Ascend/GPU/CPU|线性代数函数
|[mindspore.ops.floor_divide](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.floor_divide.html#mindspore.ops.floor_divide)|New|按元素将第一个输入Tensor除以第二个输入Tensor，并向下取整。|r2.1: Ascend/GPU/CPU|逐元素运算
