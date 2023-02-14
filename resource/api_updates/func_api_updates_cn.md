# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
|[mindspore.ops.argwhere](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.argwhere.html#mindspore.ops.argwhere)|新增|返回一个Tensor，包含所有输入Tensor非零数值的位置。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.bincount](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bincount.html#mindspore.ops.bincount)|新增|计算非负整数数组中每个值的出现次数。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.block_diag](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.block_diag.html#mindspore.ops.block_diag)|新增|基于输入Tensor创建块对角矩阵。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.cartesian_prod](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cartesian_prod.html#mindspore.ops.cartesian_prod)|新增|对给定Tensor序列计算Cartesian乘积，类似于Python里的 itertools.product 。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.cat](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cat.html#mindspore.ops.cat)|新增|在指定轴上拼接输入Tensor。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.chunk](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.chunk.html#mindspore.ops.chunk)|新增|根据指定的轴将输入Tensor切分成块。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.column_stack](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.column_stack.html#mindspore.ops.column_stack)|新增|将多个1-D 或2-D Tensor沿着水平方向堆叠成一个2-D Tensor，即按列拼接。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.cross](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cross.html#mindspore.ops.cross)|新增|返回沿着维度 dim 上，input 和 other 的向量积（叉积）。|master: Ascend/CPU|Array操作|
|[mindspore.ops.diag](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.diag.html#mindspore.ops.diag)|新增|用给定的对角线值构造对角线Tensor。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.diagflat](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.diagflat.html#mindspore.ops.diagflat)|新增|创建一个二维Tensor，用展开后的输入作为它的对角线。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.diagonal](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.diagonal.html#mindspore.ops.diagonal)|新增|返回 input 特定的对角线视图。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.dsplit](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.dsplit.html#mindspore.ops.dsplit)|新增|沿着第三轴将输入Tensor分割成多个子Tensor。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.dyn_shape](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.dyn_shape.html#mindspore.ops.dyn_shape)|新增|返回输入Tensor的Shape。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.einsum](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.einsum.html#mindspore.ops.einsum)|新增|基于爱因斯坦求和约定（Einsum）符号，指定维度对输入Tensor元素的乘积求和。|master: GPU|Array操作|
|[mindspore.ops.hsplit](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.hsplit.html#mindspore.ops.hsplit)|新增|水平地将输入Tensor分割成多个子Tensor。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.hstack](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.hstack.html#mindspore.ops.hstack)|新增|将多个Tensor沿着水平方向进行堆叠。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.index_fill](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.index_fill.html#mindspore.ops.index_fill)|新增|按 index 中给定的顺序选择索引，将输入 value 值填充到输入Tensor x 的所有 axis 维元素。|master: GPU|Array操作|
|[mindspore.ops.index_select](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.index_select.html#mindspore.ops.index_select)|新增|返回一个新的Tensor，该Tensor沿维度 axis 按 index 中给定的顺序对 x 进行选择。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.inplace_add](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.inplace_add.html#mindspore.ops.inplace_add)|修改|根据 indices，将 x 中的对应位置加上 v 。|r2.0.0-alpha: Ascend/CPU => master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.moveaxis](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.moveaxis.html#mindspore.ops.moveaxis)|新增|将 x 在 source 中位置的维度移动到 destination 中的位置。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.movedim](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.movedim.html#mindspore.ops.movedim)|新增|将 x 在 source 中位置的维度移动到 destination 中的位置。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.nan_to_num](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.nan_to_num.html#mindspore.ops.nan_to_num)|新增|将 x 中的 NaN 、正无穷大和负无穷大值分别替换为 nan, posinf, 和 neginf 指定的值。|master: CPU|Array操作|
|[mindspore.ops.nansum](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.nansum.html#mindspore.ops.nansum)|新增|计算 x 指定维度元素的和，将非数字(NaNs)处理为零。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.nonzero](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.nonzero.html#mindspore.ops.nonzero)|新增|计算x中非零元素的下标。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.normal](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.normal.html#mindspore.ops.normal)|新增|根据正态（高斯）随机数分布生成随机数。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.renorm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.renorm.html#mindspore.ops.renorm)|新增|沿维度 dim 重新规范输入 input_x 的子张量，并且每个子张量的p范数不超过给定的最大范数 maxnorm 。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.repeat_elements](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.repeat_elements.html#mindspore.ops.repeat_elements)|新增|在指定轴上复制输入Tensor的元素，类似 np.repeat 的功能。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.repeat_interleave](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.repeat_interleave.html#mindspore.ops.repeat_interleave)|新增|沿着轴重复Tensor的元素，类似 numpy.Repeat。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.scatter](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.scatter.html#mindspore.ops.scatter)|新增|根据指定索引将 src 中的值更新到 x 中返回输出。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.sequence_mask](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.sequence_mask.html#mindspore.ops.sequence_mask)|新增|返回一个表示每个单元的前N个位置的掩码Tensor，内部元素数据类型为bool。|master: GPU/CPU|Array操作|
|[mindspore.ops.space_to_batch_nd](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.space_to_batch_nd.html#mindspore.ops.space_to_batch_nd)|修改|将空间维度划分为对应大小的块，然后在批次维度重排Tensor。|r2.0.0-alpha: Ascend/GPU/CPU => master: Ascend/CPU|Array操作|
|[mindspore.ops.sparse_segment_mean](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.sparse_segment_mean.html#mindspore.ops.sparse_segment_mean)|新增|计算输出Tensor \(output_i = \frac{\sum_j x_{indices[j]}}{N}\) ，其中平均是对所有满足 \(segment\_ids[j] == i\) 的元素， \(N\) 表示相加的元素个数。|master: GPU/CPU|Array操作|
|[mindspore.ops.sum](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.sum.html#mindspore.ops.sum)|新增|计算Tensor指定维度元素的和。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.swapaxes](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.swapaxes.html#mindspore.ops.swapaxes)|新增|交换Tensor的两个维度。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.swapdims](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.swapdims.html#mindspore.ops.swapdims)|新增|交换Tensor的两个维度。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.tensor_scatter_elements](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.tensor_scatter_elements.html#mindspore.ops.tensor_scatter_elements)|新增|根据索引逐元素更新输入Tensor的值。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.tensor_split](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.tensor_split.html#mindspore.ops.tensor_split)|新增|根据指定的轴将输入Tensor进行分割成多个子Tensor。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.tril](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.tril.html#mindspore.ops.tril)|新增|返回单个矩阵（二维Tensor）或批次输入矩阵的下三角形部分，其他位置的元素将被置零。|master: GPU/CPU|Array操作|
|[mindspore.ops.unique_consecutive](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.unique_consecutive.html#mindspore.ops.unique_consecutive)|新增|对输入Tensor中连续且重复的元素去重。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.unsorted_segment_prod](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.unsorted_segment_prod.html#mindspore.ops.unsorted_segment_prod)|修改|沿分段计算输入Tensor元素的乘积。|r2.0.0-alpha: Ascend/GPU/CPU => master: Ascend/GPU|Array操作|
|[mindspore.ops.view_as_real](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.view_as_real.html#mindspore.ops.view_as_real)|新增|将复数Tensor看作实数Tensor。|master: GPU/CPU|Array操作|
|[mindspore.ops.vsplit](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.vsplit.html#mindspore.ops.vsplit)|新增|垂直地将输入Tensor分割成多个子Tensor。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.where](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.where.html#mindspore.ops.where)|新增|返回一个Tensor，Tensor的元素从 x 或 y 中根据 condition 选择。|master: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.coo_add](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.coo_add.html#mindspore.ops.coo_add)|新增|两个COOTensor相加，根据相加的结果与 thresh 返回新的COOTensor。|master: GPU/CPU|COO函数|
|[mindspore.ops.coo_concat](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.coo_concat.html#mindspore.ops.coo_concat)|新增|根据指定的轴concat_dim对输入的COO Tensor（sp_input）进行合并操作。|master: CPU|COO函数|
|[mindspore.ops.sparse_add](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.sparse_add.html#mindspore.ops.sparse_add)|删除|两个COOTensor相加，根据相加的结果与 thresh 返回新的COOTensor。|CPU/GPU|COO函数|
|[mindspore.ops.csr_mm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.csr_mm.html#mindspore.ops.csr_mm)|新增|返回稀疏矩阵a与稀疏矩阵或稠密矩阵b的矩阵乘法结果。|master: GPU|CSR函数|
|[mindspore.ops.scatter_mul](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.scatter_mul.html#mindspore.ops.scatter_mul)|修改|根据指定更新值和输入索引通过乘法运算更新输入数据的值。|r2.0.0-alpha: Ascend/GPU/CPU => master: Ascend/CPU|Parameter操作函数|
|[mindspore.ops.scatter_nd_min](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.scatter_nd_min.html#mindspore.ops.scatter_nd_min)|新增|对Tensor中的单个值或切片应用sparse minimum。|master: Ascend/GPU/CPU|Parameter操作函数|
|[mindspore.ops.all](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.all.html#mindspore.ops.all)|新增|默认情况下，通过对维度中所有元素进行“逻辑与”来减少 x 的维度。|master: Ascend/GPU/CPU|Reduction函数|
|[mindspore.ops.any](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.any.html#mindspore.ops.any)|新增|默认情况下，通过对维度中所有元素进行“逻辑或”来减少 x 的维度。|master: Ascend/GPU/CPU|Reduction函数|
|[mindspore.ops.cummax](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cummax.html#mindspore.ops.cummax)|新增|返回一个元组（最值、索引），其中最值是输入Tensor x 沿维度 axis 的累积最大值，索引是每个最大值的索引位置。|master: GPU/CPU|Reduction函数|
|[mindspore.ops.cummin](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cummin.html#mindspore.ops.cummin)|新增|返回一个元组（最值、索引），其中最值是输入Tensor x 沿维度 axis 的累积最小值，索引是每个最小值的索引位置。|master: Ascend/GPU/CPU|Reduction函数|
|[mindspore.ops.mean](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.mean.html#mindspore.ops.mean)|新增|默认情况下，移除输入所有维度，返回 x 中所有元素的平均值。|master: Ascend/GPU/CPU|Reduction函数|
|[mindspore.ops.min](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.min.html#mindspore.ops.min)|新增|在给定轴上计算输入Tensor的最小值。|master: Ascend/GPU/CPU|Reduction函数|
|[mindspore.ops.norm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.norm.html#mindspore.ops.norm)|新增|返回给定Tensor的矩阵范数或向量范数。|master: GPU/CPU|Reduction函数|
|[mindspore.ops.full](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.full.html#mindspore.ops.full)|新增|创建一个指定shape的Tensor，并用指定值填充。|master: Ascend/GPU/CPU|Tensor创建|
|[mindspore.ops.full_like](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.full_like.html#mindspore.ops.full_like)|新增|返回一个与输入相同大小的Tensor，填充 fill_value 。|master: Ascend/GPU/CPU|Tensor创建|
|[mindspore.ops.heaviside](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.heaviside.html#mindspore.ops.heaviside)|新增|计算输入中每个元素的 Heaviside 阶跃函数。|master: GPU/CPU|Tensor创建|
|[mindspore.ops.logspace](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.logspace.html#mindspore.ops.logspace)|新增|返回按照log scale平均分布的一组数值。|master: Ascend/GPU/CPU|Tensor创建|
|[mindspore.ops.zeros](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.zeros.html#mindspore.ops.zeros)|新增|创建一个填满0的Tensor，shape由 shape 决定， dtype由 dtype 决定。|master: Ascend/GPU/CPU|Tensor创建|
|[mindspore.ops.zeros_like](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.zeros_like.html#mindspore.ops.zeros_like)|新增|创建一个填满0的Tensor，shape由 x 决定，dtype由 dtype 决定。|master: Ascend/GPU/CPU|Tensor创建|
|[mindspore.ops.affine_grid](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.affine_grid.html#mindspore.ops.affine_grid)|新增|给定一批仿射矩阵 theta，生成 2D 或 3D 流场（采样网格）。|master: GPU|图像函数|
|[mindspore.ops.col2im](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.col2im.html#mindspore.ops.col2im)|新增|将一组滑动局部块组合成一个大的Tensor。|master: GPU/CPU|图像函数|
|[mindspore.ops.interpolate](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.interpolate.html#mindspore.ops.interpolate)|新增|按照给定的 size 或 scale_factor 根据 mode 设置的插值方式，对输入 x 调整大小。|master: |图像函数|
|[mindspore.ops.pad](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.pad.html#mindspore.ops.pad)|修改|根据参数 padding 对输入进行填充。|r2.0.0-alpha: Ascend/GPU/CPU => master: GPU/CPU|图像函数|
|[mindspore.ops.cosine_embedding_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cosine_embedding_loss.html#mindspore.ops.cosine_embedding_loss)|新增|余弦相似度损失函数，用于测量两个Tensor之间的相似性。|master: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.cross_entropy](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cross_entropy.html#mindspore.ops.cross_entropy)|新增|获取预测值和目标值之间的交叉熵损失。|master: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.huber_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.huber_loss.html#mindspore.ops.huber_loss)|新增|huber_loss计算预测值和目标值之间的误差。|master: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.l1_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.l1_loss.html#mindspore.ops.l1_loss)|新增|l1_loss用于计算预测值和目标值之间的平均绝对误差。|master: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.mse_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.mse_loss.html#mindspore.ops.mse_loss)|新增|计算预测值和标签值之间的均方误差。|master: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.multi_label_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.multi_label_margin_loss.html#mindspore.ops.multi_label_margin_loss)|删除|用于优化多标签分类问题的铰链损失。|Ascend/GPU|损失函数|
|[mindspore.ops.multi_margin_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.multi_margin_loss.html#mindspore.ops.multi_margin_loss)|新增|用于优化多类分类问题的铰链损失。|master: Ascend/CPU|损失函数|
|[mindspore.ops.multilabel_margin_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.multilabel_margin_loss.html#mindspore.ops.multilabel_margin_loss)|新增|用于优化多标签分类问题的铰链损失。|master: Ascend|损失函数|
|[mindspore.ops.multilabel_soft_margin_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.multilabel_soft_margin_loss.html#mindspore.ops.multilabel_soft_margin_loss)|新增|基于最大熵计算用于多标签优化的损失。|master: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.triplet_margin_loss](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.triplet_margin_loss.html#mindspore.ops.triplet_margin_loss)|新增|三元组损失函数。|r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: master: GPU|损失函数|
|[mindspore.ops.argsort](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.argsort.html#mindspore.ops.argsort)|新增|返回输入Tensor沿轴按特定顺序排序索引。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.is_complex](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.is_complex.html#mindspore.ops.is_complex)|新增|如果Tensor的数据类型是复数，则返回True，否则返回False。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.isclose](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.isclose.html#mindspore.ops.isclose)|新增|返回一个布尔型Tensor，表示 x1 的每个元素与 x2 的对应元素在给定容忍度内是否“接近”。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.isinf](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.isinf.html#mindspore.ops.isinf)|修改|确定输入Tensor每个位置上的元素是否为无穷大或无穷小。|r2.0.0-alpha: Ascend/GPU/CPU => master: GPU/CPU|比较函数|
|[mindspore.ops.isneginf](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.isneginf.html#mindspore.ops.isneginf)|新增|逐元素判断是否是负inf。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.isposinf](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.isposinf.html#mindspore.ops.isposinf)|新增|逐元素判断是否是正inf。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.isreal](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.isreal.html#mindspore.ops.isreal)|新增|逐元素判断是否为实数。|master: GPU/CPU|比较函数|
|[mindspore.ops.lt](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.lt.html#mindspore.ops.lt)|新增|mindspore.ops.less() 的别名。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.minimum](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.minimum.html#mindspore.ops.minimum)|新增|逐元素计算两个输入Tensor中的最小值。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.msort](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.msort.html#mindspore.ops.msort)|新增|将输入Tensor的元素沿其第一维按值升序排序。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.not_equal](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.not_equal.html#mindspore.ops.not_equal)|新增|ops.ne()的别名。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.searchsorted](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.searchsorted.html#mindspore.ops.searchsorted)|新增|返回位置索引，根据这个索引将 values 插入 sorted_sequence 后，sorted_sequence 的最内维度的顺序保持不变。|master: CPU|比较函数|
|[mindspore.ops.top_k](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.top_k.html#mindspore.ops.top_k)|删除|沿最后一个维度查找 k 个最大元素和对应的索引。|Ascend/GPU/CPU|比较函数|
|[mindspore.ops.topk](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.topk.html#mindspore.ops.topk)|新增|沿给定维度查找 k 个最大或最小元素和对应的索引。|master: Ascend/GPU/CPU|比较函数|
|[mindspore.ops.celu](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.celu.html#mindspore.ops.celu)|新增|celu激活函数，逐元素计算输入Tensor的celu（Continuously differentiable exponential linear units）值。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.hardtanh](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.hardtanh.html#mindspore.ops.hardtanh)|新增|逐元素元素计算hardtanh激活函数。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.leaky_relu](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.leaky_relu.html#mindspore.ops.leaky_relu)|新增|leaky_relu激活函数。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.logsigmoid](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.logsigmoid.html#mindspore.ops.logsigmoid)|新增|按元素计算logsigmoid激活函数。|master: Ascend/GPU|激活函数|
|[mindspore.ops.prelu](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.prelu.html#mindspore.ops.prelu)|修改|带参数的线性修正单元激活函数（Parametric Rectified Linear Unit activation function）。|r2.0.0-alpha: Ascend/GPU/CPU => master: Ascend/GPU|激活函数|
|[mindspore.ops.rrelu](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.rrelu.html#mindspore.ops.rrelu)|新增|Randomized Leaky ReLU激活函数。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.silu](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.silu.html#mindspore.ops.silu)|新增|按输入逐元素计算激活函数SiLU（Sigmoid Linear Unit）。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.softmin](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.softmin.html#mindspore.ops.softmin)|新增|在指定轴上对输入Tensor执行Softmin函数做归一化操作。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.threshold](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.threshold.html#mindspore.ops.threshold)|新增|threshold激活函数，按元素计算输出。|master: Ascend/GPU/CPU|激活函数|
|[mindspore.ops.adaptive_avg_pool1d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.adaptive_avg_pool1d.html#mindspore.ops.adaptive_avg_pool1d)|新增|对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应平均池化操作。|master: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.adaptive_avg_pool3d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.adaptive_avg_pool3d.html#mindspore.ops.adaptive_avg_pool3d)|新增|对由多个平面组成的的输入Tensor，进行三维的自适应平均池化操作。|master: GPU/CPU|神经网络|
|[mindspore.ops.adaptive_max_pool1d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.adaptive_max_pool1d.html#mindspore.ops.adaptive_max_pool1d)|新增|对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应最大池化操作。|master: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.adaptive_max_pool2d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.adaptive_max_pool2d.html#mindspore.ops.adaptive_max_pool2d)|新增|对输入Tensor，提供二维自适应最大池化操作。|master: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.avg_pool1d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.avg_pool1d.html#mindspore.ops.avg_pool1d)|新增|在输入Tensor上应用1D平均池化，输入Tensor可以看作是由一系列1D平面组成的。|master: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.avg_pool2d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.avg_pool2d.html#mindspore.ops.avg_pool2d)|新增|在输入Tensor上应用2D平均池化，输入Tensor可以看作是由一系列2D平面组成的。|master: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.deformable_conv2d](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.deformable_conv2d.html#mindspore.ops.deformable_conv2d)|新增|给定4D的Tensor输入 x ， weight 和 offsets ，计算一个2D的可变形卷积。|master: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.is_tensor](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.is_tensor.html#mindspore.ops.is_tensor)|新增|判断输入对象是否为 mindspore.Tensor。|master: Ascend/GPU/CPU|类型转换|
|[mindspore.ops.addr](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.addr.html#mindspore.ops.addr)|新增|计算 vec1 和 vec2 的外积，并将其添加到 x 中。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.bmm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bmm.html#mindspore.ops.bmm)|新增|基于batch维度的两个Tensor的矩阵乘法。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.cholesky](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cholesky.html#mindspore.ops.cholesky)|新增|计算对称正定矩阵 \(A\) 或一批对称正定矩阵的Cholesky分解。|master: Ascend/CPU|线性代数函数|
|[mindspore.ops.cholesky_inverse](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cholesky_inverse.html#mindspore.ops.cholesky_inverse)|新增|计算对称正定矩阵的逆矩阵。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.det](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.det.html#mindspore.ops.det)|新增|mindspore.ops.matrix_determinant() 的别名。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.inner](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.inner.html#mindspore.ops.inner)|新增|计算两个1D Tensor的点积。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.inverse](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.inverse.html#mindspore.ops.inverse)|新增|计算输入矩阵的逆。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.kron](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.kron.html#mindspore.ops.kron)|新增|计算 x 和 y 的Kronecker积：\(x⊗y\) 。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.log_matrix_determinant](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.log_matrix_determinant.html#mindspore.ops.log_matrix_determinant)|新增|对一个或多个方阵行列式的绝对值取对数，返回其符号和值。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.logdet](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.logdet.html#mindspore.ops.logdet)|新增|计算方块矩阵或批量方块矩阵的对数行列式。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.matrix_band_part](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_band_part.html#mindspore.ops.matrix_band_part)|新增|将矩阵的每个中心带外的所有位置设置为0。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.matrix_determinant](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_determinant.html#mindspore.ops.matrix_determinant)|新增|计算一个或多个方阵的行列式。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.matrix_diag_part](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_diag_part.html#mindspore.ops.matrix_diag_part)|新增|返回输入Tensor的对角线部分。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.matrix_exp](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_exp.html#mindspore.ops.matrix_exp)|修改|计算方阵的矩阵指数。|r2.0.0-alpha: CPU => master: Ascend/CPU|线性代数函数|
|[mindspore.ops.matrix_power](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_power.html#mindspore.ops.matrix_power)|新增|计算一个方阵的（整数）n次幂。|master: CPU|线性代数函数|
|[mindspore.ops.matrix_set_diag](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_set_diag.html#mindspore.ops.matrix_set_diag)|新增|返回具有新的对角线值的批处理矩阵Tensor。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.matrix_solve](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_solve.html#mindspore.ops.matrix_solve)|新增|求解线性方程组。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.mm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.mm.html#mindspore.ops.mm)|新增|计算两个矩阵的乘积。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.mv](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.mv.html#mindspore.ops.mv)|新增|实现矩阵 **`mat`** 和向量 **`vec`** 相乘。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.orgqr](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.orgqr.html#mindspore.ops.orgqr)|新增|计算 Householder 矩阵乘积的前 \(N\) 列。|master: Ascend/CPU|线性代数函数|
|[mindspore.ops.outer](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.outer.html#mindspore.ops.outer)|新增|计算 x1 和 x2 的外积。|master: Ascend/GPU/CPU|线性代数函数|
|[mindspore.ops.slogdet](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.slogdet.html#mindspore.ops.slogdet)|新增|对一个或多个方阵行列式的绝对值取对数，返回其符号和值。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.svd](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd)|新增|计算单个或多个矩阵的奇异值分解。|master: GPU/CPU|线性代数函数|
|[mindspore.ops.bartlett_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bartlett_window.html#mindspore.ops.bartlett_window)|修改|巴特利特窗口函数。|r2.0.0-alpha: GPU/CPU => master: Ascend/GPU/CPU|谱函数|
|[mindspore.ops.blackman_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.blackman_window.html#mindspore.ops.blackman_window)|修改|布莱克曼窗口函数。|r2.0.0-alpha: GPU/CPU => master: Ascend/GPU/CPU|谱函数|
|[mindspore.ops.hamming_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.hamming_window.html#mindspore.ops.hamming_window)|新增|返回一个Hamming window。|master: Ascend/GPU/CPU|谱函数|
|[mindspore.ops.hann_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.hann_window.html#mindspore.ops.hann_window)|新增|生成一个Hann window。|master: Ascend/GPU/CPU|谱函数|
|[mindspore.ops.kaiser_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.kaiser_window.html#mindspore.ops.kaiser_window)|新增|生成一个Kaiser window，也叫做Kaiser-Bessel window。|master: Ascend/GPU/CPU|谱函数|
|[mindspore.ops.dist](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.dist.html#mindspore.ops.dist)|新增|计算输入中每对行向量之间的p-范数距离。|master: Ascend/GPU/CPU|距离函数|
|[mindspore.ops.accumulate_n](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.accumulate_n.html#mindspore.ops.accumulate_n)|新增|逐元素将所有输入的Tensor相加。|master: Ascend|逐元素运算|
|[mindspore.ops.addmv](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.addmv.html#mindspore.ops.addmv)|新增|mat 和 vec 相乘，且将输入向量 x 加到最终结果中。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.arcsinh](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.arcsinh.html#mindspore.ops.arcsinh)|新增|mindspore.ops.asinh() 的别名。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.arctanh](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.arctanh.html#mindspore.ops.arctanh)|新增|mindspore.ops.atanh() 的别名。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.bitwise_left_shift](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bitwise_left_shift.html#mindspore.ops.bitwise_left_shift)|新增|对输入 x 进行左移 other 位运算。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.bitwise_right_shift](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bitwise_right_shift.html#mindspore.ops.bitwise_right_shift)|新增|对输入 x 进行右移 other 位运算。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.clamp](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.clamp.html#mindspore.ops.clamp)|新增|将输入Tensor的值裁剪到指定的最小值和最大值之间。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.clip](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.clip.html#mindspore.ops.clip)|新增|mindspore.ops.clamp() 的别名。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.copysign](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.copysign.html#mindspore.ops.copysign)|新增|逐元素地创建一个新的浮点Tensor，其大小为 x，符号为 other 的符号。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.cosine_similarity](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cosine_similarity.html#mindspore.ops.cosine_similarity)|新增|沿轴计算的x1和x2之间的余弦相似度。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.cov](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cov.html#mindspore.ops.cov)|新增|给定输入 x 和权重，估计输入 x 的协方差矩阵，其中输入的行是变量，列是观察值。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.diff](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.diff.html#mindspore.ops.diff)|新增|沿着给定维度计算输入Tensor的n阶前向差分。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.digamma](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.digamma.html#mindspore.ops.digamma)|新增|计算gamma对数函数在输入上的梯度。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.exp2](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.exp2.html#mindspore.ops.exp2)|新增|逐元素计算Tensor x 以2为底的指数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.float_power](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power)|新增|计算 x 的指数幂。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.fmod](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.fmod.html#mindspore.ops.fmod)|新增|计算除法运算 x/other 的浮点余数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.frac](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.frac.html#mindspore.ops.frac)|新增|计算 x 中每个元素的小数部分。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.gcd](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.gcd.html#mindspore.ops.gcd)|新增|按元素计算输入Tensor的最大公约数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.hypot](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.hypot.html#mindspore.ops.hypot)|新增|按元素计算以输入Tensor为直角边的三角形的斜边。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.i0](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.i0.html#mindspore.ops.i0)|修改|mindspore.ops.bessel_i0() 的别名。|r2.0.0-alpha: Ascend/GPU/CPU => master: GPU/CPU|逐元素运算|
|[mindspore.ops.igamma](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.igamma.html#mindspore.ops.igamma)|新增|计算正规化的下层不完全伽马函数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.igammac](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.igammac.html#mindspore.ops.igammac)|新增|计算正规化的上层不完全伽马函数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.lcm](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.lcm.html#mindspore.ops.lcm)|修改|逐元素计算两个输入Tensor的最小公倍数。|r2.0.0-alpha: GPU/CPU => master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.ldexp](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ldexp.html#mindspore.ops.ldexp)|新增|将输入乘以 \(2^{other}\) 。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.lgamma](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.lgamma.html#mindspore.ops.lgamma)|新增|计算输入的绝对值的gamma函数的自然对数。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.logaddexp](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.logaddexp.html#mindspore.ops.logaddexp)|新增|计算输入的指数和的对数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.logaddexp2](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.logaddexp2.html#mindspore.ops.logaddexp2)|新增|计算以2为底的输入的指数和的对数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.logical_xor](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.logical_xor.html#mindspore.ops.logical_xor)|新增|逐元素计算两个Tensor的逻辑异或运算。|master: CPU|逐元素运算|
|[mindspore.ops.mvlgamma](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.mvlgamma.html#mindspore.ops.mvlgamma)|新增|逐元素计算 p 维多元对数伽马函数值。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.ravel](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ravel.html#mindspore.ops.ravel)|新增|返回一个展开的一维Tensor。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.real](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.real.html#mindspore.ops.real)|新增|返回输入Tensor的实数部分。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.reciprocal](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.reciprocal.html#mindspore.ops.reciprocal)|新增|返回输入Tensor的倒数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.remainder](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.remainder.html#mindspore.ops.remainder)|新增|逐元素计算第一个元素除第二个元素的余数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.rot90](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.rot90.html#mindspore.ops.rot90)|新增|沿轴指定的平面内将n-D Tensor旋转90度。|master: Ascend/GPU|逐元素运算|
|[mindspore.ops.rsqrt](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.rsqrt.html#mindspore.ops.rsqrt)|新增|逐元素计算输入Tensor元素的平方根倒数。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.sgn](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.sgn.html#mindspore.ops.sgn)|新增|此方法为 mindspore.ops.sign() 在复数Tensor上的扩展。|master: GPU/CPU|逐元素运算|
|[mindspore.ops.sign](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.sign.html#mindspore.ops.sign)|新增|按sign公式逐元素计算输入Tensor。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.signbit](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.signbit.html#mindspore.ops.signbit)|新增|在符号位已设置（小于零）的情况下，按元素位置返回True。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.sinc](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.sinc.html#mindspore.ops.sinc)|新增|按照以下公式逐元素计算输入Tensor的数学正弦函数。|master: CPU|逐元素运算|
|[mindspore.ops.t](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.t.html#mindspore.ops.t)|新增|转置二维Tensor。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.tanhshrink](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.tanhshrink.html#mindspore.ops.tanhshrink)|新增|按元素计算 \(Tanhshrink(x)=x-Tanh(x)\) 。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.trapz](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.trapz.html#mindspore.ops.trapz)|新增|使用梯形法则沿给定轴对 y (x)进行积分。|master: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.bernoulli](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bernoulli.html#mindspore.ops.bernoulli)|修改|以p的概率随机将输出的元素设置为0或1，服从伯努利分布。|r2.0.0-alpha: GPU => master: GPU/CPU|随机生成函数|
|[mindspore.ops.laplace](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.laplace.html#mindspore.ops.laplace)|修改|根据拉普拉斯分布生成随机数。|r2.0.0-alpha: Ascend => master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.rand](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.rand.html#mindspore.ops.rand)|新增|返回一个Tensor，shape和dtype由输入决定，其元素为服从均匀分布的 \([0, 1)\) 区间的数字。|master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.rand_like](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.rand_like.html#mindspore.ops.rand_like)|新增|返回一个Tensor，shape和dtype由输入决定，其元素为服从均匀分布的 \([0, 1)\) 区间的数字。|master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.randint](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.randint.html#mindspore.ops.randint)|新增|返回一个Tensor，其元素为 [ low , high ) 区间的随机整数。|master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.randint_like](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.randint_like.html#mindspore.ops.randint_like)|新增|返回一个Tensor，其元素为 [ low , high ) 区间的随机整数。|master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.randn](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.randn.html#mindspore.ops.randn)|新增|返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的 \([0, 1)\) 区间的数字。|master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.randn_like](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.randn_like.html#mindspore.ops.randn_like)|新增|返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的 \([0, 1)\) 区间的数字。|master: Ascend/GPU/CPU|随机生成函数|
|[mindspore.ops.random_gamma](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.random_gamma.html#mindspore.ops.random_gamma)|新增|根据伽马分布产生成随机数。|master: CPU|随机生成函数|
|[mindspore.ops.random_poisson](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.random_poisson.html#mindspore.ops.random_poisson)|新增|从一个指定均值为 rate 的泊松分布中，随机生成形状为 shape 的随机数Tensor。|master: CPU|随机生成函数|
