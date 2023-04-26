# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.argwhere](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.argwhere.html#mindspore.ops.argwhere)|New|返回一个Tensor，包含所有输入Tensor非零数值的位置。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.bincount](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bincount.html#mindspore.ops.bincount)|New|统计 input 中每个值的出现次数。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.block_diag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.block_diag.html#mindspore.ops.block_diag)|New|基于输入Tensor创建块对角矩阵。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.cat](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cat.html#mindspore.ops.cat)|New|在指定轴上拼接输入Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.channel_shuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.channel_shuffle.html#mindspore.ops.channel_shuffle)|New|将shape为 \((*, C, H, W)\) 的Tensor的通道划分成 \(g\) 组，并按如下方式重新排列 \((*, \frac{C}{g}, g, H*W)\) ，。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.chunk](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.chunk.html#mindspore.ops.chunk)|New|沿着指定轴 axis 将输入Tensor切分成 chunks 个sub-tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.column_stack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.column_stack.html#mindspore.ops.column_stack)|New|将多个1-D 或2-D Tensor沿着水平方向堆叠成一个2-D Tensor，即按列拼接。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.deepcopy](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.deepcopy.html#mindspore.ops.deepcopy)|New|返回输入Tensor的深拷贝。|r2.0: |Array操作
[mindspore.ops.diag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diag.html#mindspore.ops.diag)|Changed|用给定的对角线值构造对角线Tensor。|r2.0.0-alpha: Ascend/GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.diagflat](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diagflat.html#mindspore.ops.diagflat)|New|创建一个二维Tensor，用展开后的 input 作为它的对角线。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.dsplit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dsplit.html#mindspore.ops.dsplit)|New|沿着第三轴将输入Tensor分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.dstack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dstack.html#mindspore.ops.dstack)|New|将多个Tensor沿着第三维度进行堆叠。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.dyn_shape](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dyn_shape.html#mindspore.ops.dyn_shape)|New|返回输入Tensor的shape。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.einsum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.einsum.html#mindspore.ops.einsum)|New|基于爱因斯坦求和约定（Einsum）符号，沿着指定维度对输入Tensor元素的乘积求和。|r2.0: GPU|Array操作
[mindspore.ops.flip](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.flip.html#mindspore.ops.flip)|Changed|沿给定轴翻转Tensor中元素的顺序。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.fliplr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fliplr.html#mindspore.ops.fliplr)|Changed|r2.0.0-alpha: 沿左右方向翻转Tensor中每行的元素。 => r2.0: 将输入Tensor中每一行的元素沿左右进行翻转，但保持矩阵的列不变。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.flipud](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.flipud.html#mindspore.ops.flipud)|Changed|r2.0.0-alpha: 沿上下方向翻转Tensor中每行的元素。 => r2.0: 将输入Tensor中每一列的元素沿上下进行翻转，但保持矩阵的行不变。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.hsplit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hsplit.html#mindspore.ops.hsplit)|New|水平地将输入Tensor分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.hstack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hstack.html#mindspore.ops.hstack)|New|将多个Tensor沿着水平方向进行堆叠。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.index_fill](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.index_fill.html#mindspore.ops.index_fill)|Changed|r2.0.0-alpha: 按 index 中给定的顺序选择索引，将输入 value 值填充到输入Tensor x 的所有 dim 维元素。 => r2.0: 按 index 中给定的顺序选择索引，将输入 value 值填充到输入Tensor x 的所有 axis 维元素。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.index_select](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.index_select.html#mindspore.ops.index_select)|New|返回一个新的Tensor，该Tensor沿维度 axis 按 index 中给定的索引对 input 进行选择。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.inplace_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_add.html#mindspore.ops.inplace_add)|Changed|根据 indices，将 x 中的对应位置加上 v 。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.inplace_index_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_index_add.html#mindspore.ops.inplace_index_add)|New|逐元素将一个Tensor updates 添加到原Tensor var 的指定轴和索引处。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.inplace_sub](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_sub.html#mindspore.ops.inplace_sub)|Changed|将 v 依照索引 indices 从 x 中减去。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.inplace_update](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_update.html#mindspore.ops.inplace_update)|Changed|根据 indices，将 x 中的某些值更新为 v。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: GPU/CPU|Array操作
[mindspore.ops.moveaxis](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.moveaxis.html#mindspore.ops.moveaxis)|New|将 x 在 source 中位置的维度移动到 destination 中的位置。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.movedim](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.movedim.html#mindspore.ops.movedim)|New|调换 x 中 source 和 destination 两个维度的位置。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.nan_to_num](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nan_to_num.html#mindspore.ops.nan_to_num)|New|将 input 中的 NaN 、正无穷大和负无穷大值分别替换为 nan 、posinf 和 neginf 指定的值。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.nansum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nansum.html#mindspore.ops.nansum)|New|计算 input 指定维度元素的和，将非数字(NaNs)处理为零。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.scatter](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.scatter.html#mindspore.ops.scatter)|New|根据指定索引将 src 中的值更新到 input 中返回输出。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.sequence_mask](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sequence_mask.html#mindspore.ops.sequence_mask)|Changed|返回一个表示每个单元的前N个位置的掩码Tensor，内部元素数据类型为bool。|r2.0.0-alpha: GPU => r2.0: GPU/CPU|Array操作
[mindspore.ops.sort](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sort.html#mindspore.ops.sort)|New|按指定顺序对输入Tensor的指定维上的元素进行排序。|r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.sum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sum.html#mindspore.ops.sum)|New|计算Tensor指定维度元素的和。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.swapaxes](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.swapaxes.html#mindspore.ops.swapaxes)|New|交换Tensor的两个维度。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.swapdims](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.swapdims.html#mindspore.ops.swapdims)|New|交换Tensor的两个维度。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.tensor_split](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tensor_split.html#mindspore.ops.tensor_split)|New|根据指定的轴将输入Tensor进行分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.tril](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tril.html#mindspore.ops.tril)|New|返回输入Tensor input 的下三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.unique_consecutive](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unique_consecutive.html#mindspore.ops.unique_consecutive)|Changed|对输入Tensor中连续且重复的元素去重。|r2.0.0-alpha: Ascend/GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.view_as_real](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.view_as_real.html#mindspore.ops.view_as_real)|New|将复数Tensor看作实数Tensor。|r2.0: GPU/CPU|Array操作
[mindspore.ops.vsplit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.vsplit.html#mindspore.ops.vsplit)|New|根据 indices_or_sections 将输入Tensor input 垂直分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.vstack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.vstack.html#mindspore.ops.vstack)|New|将多个Tensor沿着竖直方向进行堆叠。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.where](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.where.html#mindspore.ops.where)|New|返回一个Tensor，Tensor的元素从 x 或 y 中根据 condition 选择。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.coo_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_add.html#mindspore.ops.coo_add)|New|两个COOTensor相加，根据相加的结果与 thresh 返回新的COOTensor。|r2.0: GPU/CPU|COO函数
[mindspore.ops.coo_concat](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_concat.html#mindspore.ops.coo_concat)|New|根据指定的轴concat_dim对输入的COO Tensor（sp_input）进行合并操作。|r2.0: CPU|COO函数
[mindspore.ops.sparse_add](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.sparse_add.html#mindspore.ops.sparse_add)|Deleted|两个COOTensor相加，根据相加的结果与 thresh 返回新的COOTensor。|CPU/GPU|COO函数
[mindspore.ops.csr_mm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_mm.html#mindspore.ops.csr_mm)|New|返回稀疏矩阵a与稀疏矩阵或稠密矩阵b的矩阵乘法结果。|r2.0: GPU|CSR函数
[mindspore.ops.all](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.all.html#mindspore.ops.all)|New|默认情况下，通过对维度中所有元素进行“逻辑与”来减少 input 的维度。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.aminmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.aminmax.html#mindspore.ops.aminmax)|New|返回输入Tensor在指定轴上的最小值和最大值。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.any](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.any.html#mindspore.ops.any)|New|默认情况下，通过对维度中所有元素进行“逻辑或”来减少 input 的维度。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.fmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fmax.html#mindspore.ops.fmax)|New|逐元素计算输入Tensor的最大值。|r2.0: CPU|Reduction函数
[mindspore.ops.histc](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.histc.html#mindspore.ops.histc)|New|计算Tensor的直方图。|r2.0: Ascend/CPU|Reduction函数
[mindspore.ops.std](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.std.html#mindspore.ops.std)|Changed|r2.0.0-alpha: 默认情况下，输出Tensor各维度上的标准差与均值，也可以对指定维度求标准差与均值。 => r2.0: 默认情况下，输出Tensor各维度上的标准差，也可以对指定维度求标准差。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.std_mean](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.std_mean.html#mindspore.ops.std_mean)|New|默认情况下，输出Tensor各维度上的标准差和均值，也可以对指定维度求标准差和均值。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.var](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.var.html#mindspore.ops.var)|New|默认情况下，输出Tensor各维度上的方差，也可以对指定维度求方差。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.var_mean](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.var_mean.html#mindspore.ops.var_mean)|New|默认情况下，输出Tensor各维度上的方差和均值，也可以对指定维度求方差和均值。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.full](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.full.html#mindspore.ops.full)|New|创建一个指定shape的Tensor，并用指定值填充。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.full_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.full_like.html#mindspore.ops.full_like)|New|返回一个shape与 input 相同并且使用 fill_value 填充的Tensor。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.logspace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logspace.html#mindspore.ops.logspace)|New|返回一个大小为 steps 的1-D Tensor，其值从 \(base^{start}\) 到 \(base^{end}\) ，以 base 为底数。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.range](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.range.html#mindspore.ops.range)|Changed|r2.0.0-alpha: 返回从 start 开始，步长为 delta ，且不超过 limit （不包括 limit ）的序列。 => r2.0: 返回从 start 开始，步长为 step ，且不超过 end （不包括 end ）的序列。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: GPU/CPU|Tensor创建
[mindspore.ops.zeros](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.zeros.html#mindspore.ops.zeros)|New|创建一个填满0的Tensor，shape由 size 决定， dtype由 dtype 决定。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.zeros_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.zeros_like.html#mindspore.ops.zeros_like)|New|创建一个填满0的Tensor，shape由 input 决定，dtype由 dtype 决定。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.col2im](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.col2im.html#mindspore.ops.col2im)|Changed|将一组滑动局部块组合成一个大的Tensor。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|r2.0.0-alpha: Array操作 => r2.0: 图像函数
[mindspore.ops.cross](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cross.html#mindspore.ops.cross)|Changed|返回沿着维度 dim 上，input 和 other 的向量积（叉积）。|r2.0.0-alpha: CPU => r2.0: Ascend/CPU|r2.0.0-alpha: 线性代数函数 => r2.0: Array操作
[mindspore.ops.heaviside](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.heaviside.html#mindspore.ops.heaviside)|Changed|r2.0.0-alpha: 计算输入中每​​个元素的 Heaviside 阶跃函数。 => r2.0: 计算输入中每个元素的 Heaviside 阶跃函数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|r2.0.0-alpha: 逐元素运算 => r2.0: Tensor创建
[mindspore.ops.affine_grid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.affine_grid.html#mindspore.ops.affine_grid)|New|基于输入的批量仿射矩阵 theta ，返回一个二维或三维的流场（采样网格）。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.pad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pad.html#mindspore.ops.pad)|Changed|根据参数 padding 对输入进行填充。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: GPU/CPU|图像函数
[mindspore.ops.upsample](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.upsample.html#mindspore.ops.upsample)|New|mindspore.ops.interpolate() 的别名。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.cosine_embedding_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cosine_embedding_loss.html#mindspore.ops.cosine_embedding_loss)|New|余弦相似度损失函数，用于测量两个Tensor之间的相似性。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.ctc_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ctc_loss.html#mindspore.ops.ctc_loss)|New|计算CTC（Connectist Temporal Classification）损失和梯度。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.huber_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.huber_loss.html#mindspore.ops.huber_loss)|New|计算预测值和目标值之间的误差，兼具 mindspore.ops.l1_loss() 和 mindspore.ops.mse_loss() 的优点。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.l1_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.l1_loss.html#mindspore.ops.l1_loss)|New|用于计算预测值和目标值之间的平均绝对误差。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.mse_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mse_loss.html#mindspore.ops.mse_loss)|New|计算预测值和标签值之间的均方误差。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.multi_label_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.multi_label_margin_loss.html#mindspore.ops.multi_label_margin_loss)|Deleted|用于优化多标签分类问题的铰链损失。|Ascend/GPU|损失函数
[mindspore.ops.multi_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multi_margin_loss.html#mindspore.ops.multi_margin_loss)|New|用于优化多分类问题的合页损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.multilabel_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multilabel_margin_loss.html#mindspore.ops.multilabel_margin_loss)|New|用于优化多标签分类问题的合页损失。|r2.0: Ascend/GPU|损失函数
[mindspore.ops.multilabel_soft_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multilabel_soft_margin_loss.html#mindspore.ops.multilabel_soft_margin_loss)|New|基于最大熵计算用于多标签优化的损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.triplet_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.triplet_margin_loss.html#mindspore.ops.triplet_margin_loss)|New|三元组损失函数。|r2.0: GPU|损失函数
[mindspore.ops.argsort](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.argsort.html#mindspore.ops.argsort)|New|按指定顺序对输入Tensor沿给定维度进行排序，并返回排序后的索引。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.is_complex](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.is_complex.html#mindspore.ops.is_complex)|New|如果Tensor的数据类型是复数，则返回True，否则返回False。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isneginf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isneginf.html#mindspore.ops.isneginf)|New|逐元素判断是否是负inf。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isposinf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isposinf.html#mindspore.ops.isposinf)|New|逐元素判断是否是正inf。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isreal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isreal.html#mindspore.ops.isreal)|New|逐元素判断是否为实数。|r2.0: GPU/CPU|比较函数
[mindspore.ops.lt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lt.html#mindspore.ops.lt)|New|mindspore.ops.less() 的别名。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.msort](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.msort.html#mindspore.ops.msort)|New|将输入Tensor的元素沿其第一个维度按值升序排序。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.not_equal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.not_equal.html#mindspore.ops.not_equal)|New|mindspore.ops.ne() 的别名。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.searchsorted](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.searchsorted.html#mindspore.ops.searchsorted)|New|返回位置索引，根据这个索引将 values 插入 sorted_sequence 后，sorted_sequence 的最内维度的顺序保持不变。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.top_k](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.top_k.html#mindspore.ops.top_k)|Deleted|沿最后一个维度查找 k 个最大元素和对应的索引。|Ascend/GPU/CPU|比较函数
[mindspore.ops.topk](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.topk.html#mindspore.ops.topk)|New|沿给定维度查找 k 个最大或最小元素和对应的索引。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.glu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.glu.html#mindspore.ops.glu)|Changed|门线性单元函数（Gated Linear Unit function）。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: Ascend/CPU|激活函数
[mindspore.ops.hardsigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hardsigmoid.html#mindspore.ops.hardsigmoid)|New|Hard Sigmoid激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.hardtanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hardtanh.html#mindspore.ops.hardtanh)|New|逐元素元素计算hardtanh激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.leaky_relu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.leaky_relu.html#mindspore.ops.leaky_relu)|New|leaky_relu激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.logsigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logsigmoid.html#mindspore.ops.logsigmoid)|New|按元素计算logsigmoid激活函数。|r2.0: Ascend/GPU|激活函数
[mindspore.ops.rrelu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rrelu.html#mindspore.ops.rrelu)|New|Randomized Leaky ReLU激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.silu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.silu.html#mindspore.ops.silu)|New|按输入逐元素计算激活函数SiLU（Sigmoid Linear Unit）。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.soft_shrink](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.soft_shrink.html#mindspore.ops.soft_shrink)|Deleted|Soft Shrink激活函数，按输入元素计算输出。|Ascend/CPU/GPU|激活函数
[mindspore.ops.softmin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.softmin.html#mindspore.ops.softmin)|New|在指定轴上对输入Tensor执行Softmin函数做归一化操作。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.softshrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.softshrink.html#mindspore.ops.softshrink)|New|逐元素计算Soft Shrink激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.threshold](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.threshold.html#mindspore.ops.threshold)|New|使用阈值 thr 参数对 input 逐元素阈值化，并将其结果作为Tensor返回。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.adaptive_avg_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adaptive_avg_pool1d.html#mindspore.ops.adaptive_avg_pool1d)|New|对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应平均池化操作。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.adaptive_avg_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adaptive_avg_pool3d.html#mindspore.ops.adaptive_avg_pool3d)|Changed|r2.0.0-alpha: 对由多个平面组成的的输入Tensor，进行三维的自适应平均池化操作。 => r2.0: 对一个多平面输入信号执行三维自适应平均池化。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.adaptive_max_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adaptive_max_pool1d.html#mindspore.ops.adaptive_max_pool1d)|New|对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应最大池化操作。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.adaptive_max_pool3d](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.adaptive_max_pool3d.html#mindspore.ops.adaptive_max_pool3d)|Deleted|对由多个平面组成的的输入Tensor，应用三维的自适应最大池化操作。|GPU/CPU|神经网络
[mindspore.ops.avg_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.avg_pool1d.html#mindspore.ops.avg_pool1d)|New|在输入Tensor上应用1D平均池化，输入Tensor可以看作是由一系列1D平面组成的。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.avg_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.avg_pool3d.html#mindspore.ops.avg_pool3d)|Changed|在输入Tensor上应用3D平均池化，输入Tensor可以看作是由一系列3D平面组成的。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.conv1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.conv1d.html#mindspore.ops.conv1d)|New|对输入Tensor计算一维卷积。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.ctc_greedy_decoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ctc_greedy_decoder.html#mindspore.ops.ctc_greedy_decoder)|Changed|对输入中给定的logits执行贪婪解码。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.fold](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fold.html#mindspore.ops.fold)|Changed|将提取出的滑动局部区域块还原成更大的输出Tensor。|r2.0.0-alpha: CPU/GPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.fractional_max_pool2d](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.fractional_max_pool2d.html#mindspore.ops.fractional_max_pool2d)|Deleted|对多个输入平面组成的输入上应用2D分数最大池化。|CPU|神经网络
[mindspore.ops.max_pool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_pool2d.html#mindspore.ops.max_pool2d)|New|二维最大值池化。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_pool3d.html#mindspore.ops.max_pool3d)|Changed|三维最大值池化。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_unpool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_unpool1d.html#mindspore.ops.max_unpool1d)|Changed|r2.0.0-alpha: maxpool1d 的部分逆过程。 => r2.0: max_pool1d 的逆过程。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_unpool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_unpool2d.html#mindspore.ops.max_unpool2d)|Changed|r2.0.0-alpha: maxpool2d 的部分逆过程。 => r2.0: max_pool2d 的逆过程。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_unpool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_unpool3d.html#mindspore.ops.max_unpool3d)|Changed|r2.0.0-alpha: maxpool3d 的部分逆过程。 => r2.0: mindspore.ops.max_pool3d() 的逆过程。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.unfold](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.unfold.html#mindspore.ops.unfold)|Deleted|从一个输入Tensor中，提取出滑动的局部区域块。|Ascend/CPU|神经网络
[mindspore.ops.is_tensor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.is_tensor.html#mindspore.ops.is_tensor)|New|判断输入对象是否为 mindspore.Tensor。|r2.0: Ascend/GPU/CPU|类型转换
[mindspore.ops.cholesky](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cholesky.html#mindspore.ops.cholesky)|Changed|r2.0.0-alpha: 计算对称正定矩阵 \(A\) 或一批对称正定矩阵的Cholesky分解。 => r2.0: 计算对称正定矩阵或一批对称正定矩阵的Cholesky分解。|r2.0.0-alpha: Ascend/CPU => r2.0: GPU/CPU|线性代数函数
[mindspore.ops.cholesky_inverse](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.cholesky_inverse.html#mindspore.ops.cholesky_inverse)|Deleted|计算对称正定矩阵的逆矩阵。|GPU/CPU|线性代数函数
[mindspore.ops.eig](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.eig.html#mindspore.ops.eig)|New|计算输入方阵（batch方阵）的特征值和特征向量。|r2.0: Ascend/CPU|线性代数函数
[mindspore.ops.geqrf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.geqrf.html#mindspore.ops.geqrf)|New|将矩阵分解为正交矩阵 Q 和上三角矩阵 R 的乘积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.inner](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inner.html#mindspore.ops.inner)|New|计算两个1D Tensor的点积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.inverse](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inverse.html#mindspore.ops.inverse)|New|计算输入矩阵的逆。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.kron](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.kron.html#mindspore.ops.kron)|New|计算 x 和 y 的Kronecker积：\(x ⊗ y\) 。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.logdet](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logdet.html#mindspore.ops.logdet)|New|计算方块矩阵或批量方块矩阵的对数行列式。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.lu_unpack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lu_unpack.html#mindspore.ops.lu_unpack)|New|将 LU_data 和 LU_pivots 还原为P, L, U矩阵，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.matrix_band_part](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.matrix_band_part.html#mindspore.ops.matrix_band_part)|Deleted|将矩阵的每个中心带外的所有位置设置为0。|GPU/CPU|线性代数函数
[mindspore.ops.matrix_exp](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.matrix_exp.html#mindspore.ops.matrix_exp)|Deleted|计算方阵的矩阵指数。|CPU|线性代数函数
[mindspore.ops.matrix_set_diag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.matrix_set_diag.html#mindspore.ops.matrix_set_diag)|Changed|返回具有新的对角线值的批处理矩阵Tensor。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.matrix_solve](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.matrix_solve.html#mindspore.ops.matrix_solve)|Changed|求解线性方程组。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/CPU|线性代数函数
[mindspore.ops.mm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mm.html#mindspore.ops.mm)|New|计算两个矩阵的乘积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.mv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mv.html#mindspore.ops.mv)|New|实现矩阵 mat 和向量 vec 相乘。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.orgqr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.orgqr.html#mindspore.ops.orgqr)|New|计算 mindspore.ops.Geqrf 返回的正交矩阵 \(Q\) 的显式表示。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.outer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.outer.html#mindspore.ops.outer)|New|计算 input 和 vec2 的外积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.qr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.qr.html#mindspore.ops.qr)|New|返回一个或多个矩阵的QR（正交三角）分解。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.slogdet](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.slogdet.html#mindspore.ops.slogdet)|New|对一个或多个方阵行列式的绝对值取对数，返回其符号和值。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.trace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.trace.html#mindspore.ops.trace)|New|返回input的对角线方向上的总和。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.bartlett_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bartlett_window.html#mindspore.ops.bartlett_window)|Changed|r2.0.0-alpha: 巴特利特窗口函数。 => r2.0: Bartlett窗口函数是一种三角形状的加权函数，通常用于平滑处理或频域分析信号。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.blackman_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.blackman_window.html#mindspore.ops.blackman_window)|Changed|r2.0.0-alpha: 布莱克曼窗口函数。 => r2.0: 布莱克曼窗口函数，常用来为FFT截取有限长的信号片段。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.hamming_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hamming_window.html#mindspore.ops.hamming_window)|New|返回一个Hamming window。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.hann_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hann_window.html#mindspore.ops.hann_window)|New|生成一个Hann window。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.kaiser_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.kaiser_window.html#mindspore.ops.kaiser_window)|New|生成一个Kaiser window，也叫做Kaiser-Bessel window。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.dist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dist.html#mindspore.ops.dist)|New|计算输入中每对行向量之间的 \(p\)-norm距离。|r2.0: Ascend/GPU/CPU|距离函数
[mindspore.ops.pdist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pdist.html#mindspore.ops.pdist)|Changed|计算输入中每对行向量之间的p-范数距离。|r2.0.0-alpha: CPU => r2.0: GPU/CPU|距离函数
[mindspore.ops.addmv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.addmv.html#mindspore.ops.addmv)|New|mat 和 vec 相乘，且将输入向量 x 加到最终结果中。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.angle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.angle.html#mindspore.ops.angle)|Changed|r2.0.0-alpha: 返回复数Tensor的元素参数。 => r2.0: 逐元素计算复数Tensor的辐角。|r2.0.0-alpha: CPU => r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arcsinh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arcsinh.html#mindspore.ops.arcsinh)|New|mindspore.ops.asinh() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arctanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arctanh.html#mindspore.ops.arctanh)|New|mindspore.ops.atanh() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.atleast_1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.atleast_1d.html#mindspore.ops.atleast_1d)|New|调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于1。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.atleast_2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.atleast_2d.html#mindspore.ops.atleast_2d)|New|调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于2。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.atleast_3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.atleast_3d.html#mindspore.ops.atleast_3d)|New|调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于3。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.bessel_i1e](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bessel_i1e.html#mindspore.ops.bessel_i1e)|Changed|逐元素计算并返回输入Tensor的Bessel i1e函数值。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.bitwise_left_shift](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bitwise_left_shift.html#mindspore.ops.bitwise_left_shift)|New|逐元素对输入 input 进行左移位运算, 移动的位数由 other 指定。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.bitwise_right_shift](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bitwise_right_shift.html#mindspore.ops.bitwise_right_shift)|New|逐元素对输入 input 进行右移位运算, 移动的位数由 other 指定。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.clamp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.clamp.html#mindspore.ops.clamp)|New|将输入Tensor的值裁剪到指定的最小值和最大值之间。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.clip](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.clip.html#mindspore.ops.clip)|New|mindspore.ops.clamp() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.combinations](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.combinations.html#mindspore.ops.combinations)|New|返回输入Tensor中元素的所有长度为 r 的子序列。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.cosine_similarity](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cosine_similarity.html#mindspore.ops.cosine_similarity)|New|沿轴计算的x1和x2之间的余弦相似度。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.cov](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cov.html#mindspore.ops.cov)|New|给定输入 input 和权重，返回输入 input 的协方差矩阵(每对变量的协方差的方阵)，其中输入行是变量，列是观察值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.diag_embed](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diag_embed.html#mindspore.ops.diag_embed)|New|生成一个Tensor，其对角线值由 input 中的值填充，其余位置置0。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.diff](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diff.html#mindspore.ops.diff)|New|沿着给定维度计算输入Tensor x 的n阶前向差分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.digamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.digamma.html#mindspore.ops.digamma)|New|计算lgamma对数函数在输入上的梯度。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.exp2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.exp2.html#mindspore.ops.exp2)|New|逐元素计算Tensor input 以2为底的指数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.float_power](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power)|New|计算 input 的指数幂。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.fmod](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fmod.html#mindspore.ops.fmod)|New|计算除法运算 input/other 的浮点余数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.frac](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.frac.html#mindspore.ops.frac)|New|计算 x 中每个元素的小数部分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.gcd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.gcd.html#mindspore.ops.gcd)|New|按元素计算输入Tensor的最大公约数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.hypot](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hypot.html#mindspore.ops.hypot)|Changed|按元素计算以输入Tensor为直角边的三角形的斜边。|r2.0.0-alpha: CPU => r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.i0](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.i0.html#mindspore.ops.i0)|Changed|mindspore.ops.bessel_i0() 的别名。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: GPU/CPU|逐元素运算
[mindspore.ops.imag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.imag.html#mindspore.ops.imag)|New|返回包含输入Tensor的虚部。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.lcm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lcm.html#mindspore.ops.lcm)|Changed|逐元素计算两个输入Tensor的最小公倍数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.logical_xor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logical_xor.html#mindspore.ops.logical_xor)|Changed|逐元素计算两个Tensor的逻辑异或运算。|r2.0.0-alpha: CPU => r2.0: Ascend/CPU|逐元素运算
[mindspore.ops.logit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logit.html#mindspore.ops.logit)|Changed|逐元素计算Tensor的logit值。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.mvlgamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mvlgamma.html#mindspore.ops.mvlgamma)|New|逐元素计算 p 维多元对数伽马函数值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.nextafter](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nextafter.html#mindspore.ops.nextafter)|New|逐元素返回 input 指向 other 的下一个可表示值符点值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.polar](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.polar.html#mindspore.ops.polar)|New|将极坐标转化为笛卡尔坐标。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.polygamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.polygamma.html#mindspore.ops.polygamma)|New|计算关于 input 的多伽马函数的 \(n\) 阶导数。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.ravel](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ravel.html#mindspore.ops.ravel)|New|沿着0轴方向，将多维Tensor展开成一维。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.real](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.real.html#mindspore.ops.real)|New|返回输入Tensor的实数部分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.reciprocal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.reciprocal.html#mindspore.ops.reciprocal)|New|返回输入Tensor每个元素的倒数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.roll](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.roll.html#mindspore.ops.roll)|Deleted|沿轴移动Tensor的元素。|Ascend/GPU|逐元素运算
[mindspore.ops.rot90](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rot90.html#mindspore.ops.rot90)|New|沿轴指定的平面内将n-D Tensor旋转90度。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.rsqrt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rsqrt.html#mindspore.ops.rsqrt)|New|逐元素计算输入Tensor元素的平方根倒数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sgn](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sgn.html#mindspore.ops.sgn)|New|mindspore.ops.sign() 在复数上的扩展。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sign](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sign.html#mindspore.ops.sign)|New|按sign公式逐元素计算输入Tensor。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.signbit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.signbit.html#mindspore.ops.signbit)|New|判断每个元素的符号，如果元素值小于0则对应输出的位置为True，否则为False。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sinc](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sinc.html#mindspore.ops.sinc)|New|计算输入的归一化正弦值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.t](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.t.html#mindspore.ops.t)|New|转置二维Tensor。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.tanhshrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tanhshrink.html#mindspore.ops.tanhshrink)|New|Tanhshrink激活函数， \(Tanhshrink(x)=x-Tanh(x)\) ，其中 \(x\) 即输入 input。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.trapz](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.trapz.html#mindspore.ops.trapz)|New|使用梯形法则沿给定轴 dim 对 y(x) 进行积分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.tril_indices](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tril_indices.html#mindspore.ops.tril_indices)|New|计算 row * col 行列矩阵的下三角元素的索引，并将它们作为一个 2xN 的Tensor返回。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.triu_indices](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.triu_indices.html#mindspore.ops.triu_indices)|New|计算 row * col 行列矩阵的上三角元素的索引，并将它们作为一个 2xN 的Tensor返回。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.bernoulli](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bernoulli.html#mindspore.ops.bernoulli)|Changed|r2.0.0-alpha: 以p的概率随机将输出的元素设置为0或1，服从伯努利分布。 => r2.0: 以 p 的概率随机将输出的元素设置为0或1，服从伯努利分布。|r2.0.0-alpha: GPU => r2.0: GPU/CPU|随机生成函数
[mindspore.ops.laplace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.laplace.html#mindspore.ops.laplace)|Changed|根据拉普拉斯分布生成随机数。|r2.0.0-alpha: Ascend => r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.multinomial](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multinomial.html#mindspore.ops.multinomial)|Changed|根据输入生成一个多项式分布的Tensor。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.multinomial_with_replacement](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multinomial_with_replacement.html#mindspore.ops.multinomial_with_replacement)|New|返回一个Tensor，其中每行包含从重复采样的多项式分布中抽取的 numsamples 个索引。|r2.0: CPU|随机生成函数
[mindspore.ops.rand](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rand.html#mindspore.ops.rand)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从均匀分布的 \([0, 1)\) 区间的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.rand_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rand_like.html#mindspore.ops.rand_like)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从均匀分布的 \([0, 1)\) 区间的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randint](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randint.html#mindspore.ops.randint)|New|返回一个Tensor，其元素为 [ low , high ) 区间的随机整数。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randint_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randint_like.html#mindspore.ops.randint_like)|New|返回一个Tensor，其元素为 [ low , high ) 区间的随机整数，根据 input 决定shape和dtype。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randn](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randn.html#mindspore.ops.randn)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randn_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randn_like.html#mindspore.ops.randn_like)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.random_poisson](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.random_poisson.html#mindspore.ops.random_poisson)|Changed|从一个指定均值为 rate 的泊松分布中，随机生成形状为 shape 的随机数Tensor。|r2.0.0-alpha: CPU => r2.0: GPU/CPU|随机生成函数
[mindspore.ops.randperm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randperm.html#mindspore.ops.randperm)|New|生成从 0 到 n-1 的整数随机排列。|r2.0: CPU|随机生成函数
[mindspore.ops.uniform](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.uniform.html#mindspore.ops.uniform)|Changed|生成服从均匀分布的随机数。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: GPU/CPU|随机生成函数
