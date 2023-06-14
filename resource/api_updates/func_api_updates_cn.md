# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.argwhere](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.argwhere.html#mindspore.ops.argwhere)|New|返回一个Tensor，包含所有输入Tensor非零数值的位置。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.batch_to_space_nd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.batch_to_space_nd.html#mindspore.ops.batch_to_space_nd)|Changed|用块划分批次维度，并将这些块交错回空间维度。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.bincount](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bincount.html#mindspore.ops.bincount)|New|统计 input 中每个值的出现次数。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.block_diag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.block_diag.html#mindspore.ops.block_diag)|New|基于输入Tensor创建块对角矩阵。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.cat](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cat.html#mindspore.ops.cat)|New|在指定轴上拼接输入Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.channel_shuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.channel_shuffle.html#mindspore.ops.channel_shuffle)|New|将shape为 $(\*, C, H, W)$ 的Tensor的通道划分成 $g$ 组，并按如下方式重新排列 $(\*, \frac{C}{g}, g, H*W)$ ，同时保持原始Tensor的shape不变。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.chunk](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.chunk.html#mindspore.ops.chunk)|New|沿着指定轴 axis 将输入Tensor切分成 chunks 个sub-tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.column_stack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.column_stack.html#mindspore.ops.column_stack)|New|将多个1-D 或2-D Tensor沿着水平方向堆叠成一个2-D Tensor，即按列拼接。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.conj](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.conj.html#mindspore.ops.conj)|New|逐元素计算输入Tensor的共轭。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.cross](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cross.html#mindspore.ops.cross)|New|返回沿着维度 dim 上，input 和 other 的向量积（叉积）。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.diag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diag.html#mindspore.ops.diag)|Changed|用给定的对角线值构造对角线Tensor。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.diagflat](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diagflat.html#mindspore.ops.diagflat)|New|创建一个二维Tensor，用展开后的 input 作为它的对角线。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.diagonal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diagonal.html#mindspore.ops.diagonal)|New|返回 input 特定的对角线视图。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.dsplit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dsplit.html#mindspore.ops.dsplit)|New|沿着第三轴将输入Tensor分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.dstack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dstack.html#mindspore.ops.dstack)|New|将多个Tensor沿着第三维度进行堆叠。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.dyn_shape](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dyn_shape.html#mindspore.ops.dyn_shape)|New|返回输入Tensor的shape。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.einsum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.einsum.html#mindspore.ops.einsum)|New|基于爱因斯坦求和约定（Einsum）符号，沿着指定维度对输入Tensor元素的乘积求和。|r2.0: GPU|Array操作
[mindspore.ops.flip](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.flip.html#mindspore.ops.flip)|New|沿给定轴翻转Tensor中元素的顺序。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.fliplr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fliplr.html#mindspore.ops.fliplr)|New|将输入Tensor中每一行的元素沿左右进行翻转，但保持矩阵的列不变。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.flipud](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.flipud.html#mindspore.ops.flipud)|New|将输入Tensor中每一列的元素沿上下进行翻转，但保持矩阵的行不变。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.hsplit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hsplit.html#mindspore.ops.hsplit)|New|水平地将输入Tensor分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.hstack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hstack.html#mindspore.ops.hstack)|New|将多个Tensor沿着水平方向进行堆叠。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.index_fill](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.index_fill.html#mindspore.ops.index_fill)|Changed|r1.10: 按 index 中给定的顺序选择索引，将输入 value 值填充到输入Tensor x 的所有 dim 维元素。 => r2.0: 按 index 中给定的顺序选择索引，将输入 value 值填充到输入Tensor x 的所有 axis 维元素。|r1.10: GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.index_select](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.index_select.html#mindspore.ops.index_select)|New|返回一个新的Tensor，该Tensor沿维度 axis 按 index 中给定的索引对 input 进行选择。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.inplace_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_add.html#mindspore.ops.inplace_add)|Changed|根据 indices，将 x 中的对应位置加上 v 。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.inplace_index_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_index_add.html#mindspore.ops.inplace_index_add)|New|逐元素将一个Tensor updates 添加到原Tensor var 的指定轴和索引处。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.inplace_sub](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_sub.html#mindspore.ops.inplace_sub)|Changed|将 v 依照索引 indices 从 x 中减去。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.inplace_update](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inplace_update.html#mindspore.ops.inplace_update)|Changed|根据 indices，将 x 中的某些值更新为 v。|r1.10: Ascend/GPU/CPU => r2.0: GPU/CPU|Array操作
[mindspore.ops.matrix_band_part](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.matrix_band_part.html#mindspore.ops.matrix_band_part)|Deleted|将矩阵的每个中心带外的所有位置设置为0。|GPU/CPU|Array操作
[mindspore.ops.moveaxis](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.moveaxis.html#mindspore.ops.moveaxis)|New|将 x 在 source 中位置的维度移动到 destination 中的位置。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.movedim](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.movedim.html#mindspore.ops.movedim)|New|调换 x 中 source 和 destination 两个维度的位置。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.nan_to_num](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nan_to_num.html#mindspore.ops.nan_to_num)|New|将 input 中的 NaN 、正无穷大和负无穷大值分别替换为 nan 、posinf 和 neginf 指定的值。|r2.0: Ascend/CPU|Array操作
[mindspore.ops.nansum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nansum.html#mindspore.ops.nansum)|New|计算 input 指定维度元素的和，将非数字(NaNs)处理为零。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.nonzero](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nonzero.html#mindspore.ops.nonzero)|Changed|计算x中非零元素的下标。|r1.10: GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.numel](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.numel.html#mindspore.ops.numel)|New|返回Tensor的元素的总数量。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.permute](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.permute.html#mindspore.ops.permute)|New|按照输入 axis 的维度顺序排列输入Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.repeat_interleave](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.repeat_interleave.html#mindspore.ops.repeat_interleave)|New|沿着轴重复Tensor的元素，类似 numpy.Repeat。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.reverse](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.reverse.html#mindspore.ops.reverse)|New|对输入Tensor按指定维度反转。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.scatter](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.scatter.html#mindspore.ops.scatter)|New|根据指定索引将 src 中的值更新到 input 中返回输出。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.sequence_mask](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sequence_mask.html#mindspore.ops.sequence_mask)|Changed|r1.10: 返回一个表示每个单元的前N个位置的掩码Tensor。 => r2.0: 返回一个表示每个单元的前N个位置的掩码Tensor，内部元素数据类型为bool。|r1.10: GPU => r2.0: GPU/CPU|Array操作
[mindspore.ops.shuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.shuffle.html#mindspore.ops.shuffle)|New|沿着Tensor第一维随机打乱数据。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.sort](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sort.html#mindspore.ops.sort)|New|按指定顺序对输入Tensor的指定维上的元素进行排序。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.space_to_batch_nd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.space_to_batch_nd.html#mindspore.ops.space_to_batch_nd)|Changed|将空间维度划分为对应大小的块，然后在批次维度重排Tensor。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.strided_slice](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.strided_slice.html#mindspore.ops.strided_slice)|New|对输入Tensor根据步长和索引进行切片提取。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.sum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sum.html#mindspore.ops.sum)|New|计算Tensor指定维度元素的和。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.swapaxes](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.swapaxes.html#mindspore.ops.swapaxes)|New|交换Tensor的两个维度。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.swapdims](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.swapdims.html#mindspore.ops.swapdims)|New|交换Tensor的两个维度。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.tensor_scatter_max](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tensor_scatter_max.html#mindspore.ops.tensor_scatter_max)|New|根据指定的更新值和输入索引，通过最大值运算，输出结果以Tensor形式返回。|r2.0: GPU/CPU|Array操作
[mindspore.ops.tensor_scatter_min](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tensor_scatter_min.html#mindspore.ops.tensor_scatter_min)|New|根据指定的更新值和输入索引，通过最小值运算，将结果赋值到输出Tensor中。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.tensor_split](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tensor_split.html#mindspore.ops.tensor_split)|New|根据指定的轴将输入Tensor进行分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.top_k](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.top_k.html#mindspore.ops.top_k) => [mindspore.ops.topk](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.topk.html#mindspore.ops.topk)|Changed|沿最后一个维度查找 k 个最大元素和对应的索引。|Ascend/GPU/CPU|Array操作
[mindspore.ops.tril](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tril.html#mindspore.ops.tril)|New|返回输入Tensor input 的下三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.unbind](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unbind.html#mindspore.ops.unbind)|New|根据指定轴对输入矩阵进行分解。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.unique_consecutive](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unique_consecutive.html#mindspore.ops.unique_consecutive)|Changed|对输入Tensor中连续且重复的元素去重。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.unsorted_segment_prod](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unsorted_segment_prod.html#mindspore.ops.unsorted_segment_prod)|Changed|沿分段计算输入Tensor元素的乘积。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.unsqueeze](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unsqueeze.html#mindspore.ops.unsqueeze)|New|对输入 input 在给定维上添加额外维度。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.view_as_real](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.view_as_real.html#mindspore.ops.view_as_real)|New|将复数Tensor看作实数Tensor。|r2.0: GPU/CPU|Array操作
[mindspore.ops.vsplit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.vsplit.html#mindspore.ops.vsplit)|New|根据 indices_or_sections 将输入Tensor input 垂直分割成多个子Tensor。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.vstack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.vstack.html#mindspore.ops.vstack)|New|将多个Tensor沿着竖直方向进行堆叠。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.where](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.where.html#mindspore.ops.where)|New|返回一个Tensor，Tensor的元素从 x 或 y 中根据 condition 选择。|r2.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.coo_abs](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_abs.html#mindspore.ops.coo_abs)|New|逐元素计算输入COOTensor的绝对值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_acos](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_acos.html#mindspore.ops.coo_acos)|New|逐元素计算输入COOTensor的反余弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_acosh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_acosh.html#mindspore.ops.coo_acosh)|New|逐元素计算输入COOTensor的反双曲余弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_add.html#mindspore.ops.coo_add)|New|两个COOTensor相加，根据相加的结果与 thresh 返回新的COOTensor。|r2.0: GPU/CPU|COO函数
[mindspore.ops.coo_asin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_asin.html#mindspore.ops.coo_asin)|New|逐元素计算输入COOTensor的反正弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_asinh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_asinh.html#mindspore.ops.coo_asinh)|New|计算COOTensor输入元素的反双曲正弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_atan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_atan.html#mindspore.ops.coo_atan)|New|逐元素计算输入COOTensor的反正切值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_atanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_atanh.html#mindspore.ops.coo_atanh)|New|逐元素计算输入COOTensor的反双曲正切值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_ceil](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_ceil.html#mindspore.ops.coo_ceil)|New|COOTensor向上取整函数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_concat](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_concat.html#mindspore.ops.coo_concat)|New|根据指定的轴concat_dim对输入的COO Tensor（sp_input）进行合并操作。|r2.0: CPU|COO函数
[mindspore.ops.coo_cos](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_cos.html#mindspore.ops.coo_cos)|New|逐元素计算COOTensor输入的余弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_cosh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_cosh.html#mindspore.ops.coo_cosh)|New|逐元素计算COOTensor x 的双曲余弦值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_exp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_exp.html#mindspore.ops.coo_exp)|New|逐元素计算COOTensor x 的指数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_expm1](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_expm1.html#mindspore.ops.coo_expm1)|New|逐元素计算输入COOTensor的指数，然后减去1。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_floor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_floor.html#mindspore.ops.coo_floor)|New|COOTensor逐元素向下取整函数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_inv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_inv.html#mindspore.ops.coo_inv)|New|逐元素计算输入COOTensor的倒数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_isfinite](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_isfinite.html#mindspore.ops.coo_isfinite)|New|判断COOTensor输入数据每个位置上的值是否是有限数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_isinf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_isinf.html#mindspore.ops.coo_isinf)|New|确定输入COOTensor每个位置上的元素是否为正负无穷大。|r2.0: GPU/CPU|COO函数
[mindspore.ops.coo_isnan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_isnan.html#mindspore.ops.coo_isnan)|New|判断COOTensor输入数据每个位置上的值是否是Nan。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_log](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_log.html#mindspore.ops.coo_log)|New|逐元素返回COOTensor的自然对数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_log1p](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_log1p.html#mindspore.ops.coo_log1p)|New|对输入COOTensor逐元素加一后计算自然对数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_neg](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_neg.html#mindspore.ops.coo_neg)|New|计算输入COOTensor的相反数并返回。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_relu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_relu.html#mindspore.ops.coo_relu)|New|对输入的COOTensor逐元素计算其应用ReLU激活函数后的值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_relu6](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_relu6.html#mindspore.ops.coo_relu6)|New|对输入的COOTensor计算其应用ReLU激活函数后的值，上限为6。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_round](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_round.html#mindspore.ops.coo_round)|New|对COOTensor输入数据进行四舍五入到最接近的整数数值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_sigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_sigmoid.html#mindspore.ops.coo_sigmoid)|New|Sigmoid激活函数，COOTensor逐元素计算Sigmoid激活函数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_sin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_sin.html#mindspore.ops.coo_sin)|New|逐元素计算输入COOTensor的正弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_sinh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_sinh.html#mindspore.ops.coo_sinh)|New|逐元素计算输入COOTensor的双曲正弦。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_softsign](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_softsign.html#mindspore.ops.coo_softsign)|New|COOTensor Softsign激活函数。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_sqrt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_sqrt.html#mindspore.ops.coo_sqrt)|New|逐元素返回当前COOTensor的平方根。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_square](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_square.html#mindspore.ops.coo_square)|New|逐元素返回COOTensor的平方。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_tan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_tan.html#mindspore.ops.coo_tan)|New|计算COOTensor输入元素的正切值。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.coo_tanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.coo_tanh.html#mindspore.ops.coo_tanh)|New|按元素计算COOTensor输入元素的双曲正切。|r2.0: Ascend/GPU/CPU|COO函数
[mindspore.ops.csr_abs](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_abs.html#mindspore.ops.csr_abs)|New|逐元素计算输入CSRTensor的绝对值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_acos](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_acos.html#mindspore.ops.csr_acos)|New|逐元素计算输入CSRTensor的反余弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_acosh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_acosh.html#mindspore.ops.csr_acosh)|New|逐元素计算输入CSRTensor的反双曲余弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_add](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_add.html#mindspore.ops.csr_add)|New|$a$ 和 $b$ 是CSRTensor，$alpha$ 和 $beta$ 是Tensor。|r2.0: GPU/CPU|CSR函数
[mindspore.ops.csr_asin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_asin.html#mindspore.ops.csr_asin)|New|逐元素计算输入CSRTensor的反正弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_asinh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_asinh.html#mindspore.ops.csr_asinh)|New|计算CSRTensor输入元素的反双曲正弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_atan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_atan.html#mindspore.ops.csr_atan)|New|逐元素计算输入CSRTensor的反正切值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_atanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_atanh.html#mindspore.ops.csr_atanh)|New|逐元素计算输入CSRTensor的反双曲正切值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_ceil](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_ceil.html#mindspore.ops.csr_ceil)|New|CSRTensor向上取整函数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_cos](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_cos.html#mindspore.ops.csr_cos)|New|逐元素计算CSRTensor输入的余弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_cosh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_cosh.html#mindspore.ops.csr_cosh)|New|逐元素计算CSRTensor x 的双曲余弦值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_exp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_exp.html#mindspore.ops.csr_exp)|New|逐元素计算CSRTensor x 的指数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_expm1](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_expm1.html#mindspore.ops.csr_expm1)|New|逐元素计算输入CSRTensor的指数，然后减去1。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_floor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_floor.html#mindspore.ops.csr_floor)|New|CSRTensor逐元素向下取整函数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_inv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_inv.html#mindspore.ops.csr_inv)|New|逐元素计算输入CSRTensor的倒数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_isfinite](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_isfinite.html#mindspore.ops.csr_isfinite)|New|判断CSRTensor输入数据每个位置上的值是否是有限数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_isinf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_isinf.html#mindspore.ops.csr_isinf)|New|确定输入CSRTensor每个位置上的元素是否为无穷大或无穷小。|r2.0: GPU/CPU|CSR函数
[mindspore.ops.csr_isnan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_isnan.html#mindspore.ops.csr_isnan)|New|判断CSRTensor输入数据每个位置上的值是否是Nan。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_log](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_log.html#mindspore.ops.csr_log)|New|逐元素返回CSRTensor的自然对数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_log1p](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_log1p.html#mindspore.ops.csr_log1p)|New|对输入CSRTensor逐元素加一后计算自然对数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_mm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_mm.html#mindspore.ops.csr_mm)|New|返回稀疏矩阵a与稀疏矩阵或稠密矩阵b的矩阵乘法结果。|r2.0: GPU|CSR函数
[mindspore.ops.csr_neg](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_neg.html#mindspore.ops.csr_neg)|New|计算输入CSRTensor的相反数并返回。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_relu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_relu.html#mindspore.ops.csr_relu)|New|逐元素计算CSRTensor的ReLU（Rectified Linear Unit）激活值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_relu6](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_relu6.html#mindspore.ops.csr_relu6)|New|逐元素计算CSRTensor的ReLU值，其上限为6。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_round](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_round.html#mindspore.ops.csr_round)|New|对CSRTensor输入数据进行四舍五入到最接近的整数数值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_sigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_sigmoid.html#mindspore.ops.csr_sigmoid)|New|Sigmoid激活函数，CSRTensor逐元素计算Sigmoid激活函数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_sin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_sin.html#mindspore.ops.csr_sin)|New|逐元素计算输入CSRTensor的正弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_sinh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_sinh.html#mindspore.ops.csr_sinh)|New|逐元素计算输入CSRTensor的双曲正弦。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_softmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_softmax.html#mindspore.ops.csr_softmax)|New|计算 CSRTensorMatrix 的 softmax 。|r2.0: GPU/CPU|CSR函数
[mindspore.ops.csr_softsign](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_softsign.html#mindspore.ops.csr_softsign)|New|CSRTensor Softsign激活函数。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_sqrt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_sqrt.html#mindspore.ops.csr_sqrt)|New|逐元素返回当前CSRTensor的平方根。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_square](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_square.html#mindspore.ops.csr_square)|New|逐元素返回CSRTensor的平方。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_tan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_tan.html#mindspore.ops.csr_tan)|New|逐元素计算CSRTensor的正切值。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.csr_tanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.csr_tanh.html#mindspore.ops.csr_tanh)|New|逐元素计算CSRTensor输入元素的双曲正切。|r2.0: Ascend/GPU/CPU|CSR函数
[mindspore.ops.all](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.all.html#mindspore.ops.all)|New|默认情况下，通过对维度中所有元素进行“逻辑与”来减少 input 的维度。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.aminmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.aminmax.html#mindspore.ops.aminmax)|New|返回输入Tensor在指定轴上的最小值和最大值。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.any](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.any.html#mindspore.ops.any)|New|默认情况下，通过对维度中所有元素进行“逻辑或”来减少 input 的维度。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.argmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.argmax.html#mindspore.ops.argmax)|New|返回输入Tensor在指定轴上的最大值索引。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.cumprod](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cumprod.html#mindspore.ops.cumprod)|New|返回输入的元素在 dim 维度上的累积乘积。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.cumsum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cumsum.html#mindspore.ops.cumsum)|New|计算输入Tensor x 沿轴 axis 的累积和。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.fmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fmax.html#mindspore.ops.fmax)|New|逐元素计算输入Tensor的最大值。|r2.0: CPU|Reduction函数
[mindspore.ops.histc](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.histc.html#mindspore.ops.histc)|New|计算Tensor的直方图。|r2.0: Ascend/CPU|Reduction函数
[mindspore.ops.median](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.median.html#mindspore.ops.median)|New|输出Tensor指定维度上的中值与索引。|r2.0: GPU/CPU|Reduction函数
[mindspore.ops.std](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.std.html#mindspore.ops.std)|Changed|r1.10: 默认情况下，输出Tensor各维度上的标准差与均值，也可以对指定维度求标准差与均值。 => r2.0: 默认情况下，输出Tensor各维度上的标准差，也可以对指定维度求标准差。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.std_mean](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.std_mean.html#mindspore.ops.std_mean)|New|默认情况下，输出Tensor各维度上的标准差和均值，也可以对指定维度求标准差和均值。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.var](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.var.html#mindspore.ops.var)|New|默认情况下，输出Tensor各维度上的方差，也可以对指定维度求方差。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.var_mean](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.var_mean.html#mindspore.ops.var_mean)|New|默认情况下，输出Tensor各维度上的方差和均值，也可以对指定维度求方差和均值。|r2.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.arange](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arange.html#mindspore.ops.arange)|New|返回从 start 开始，步长为 step ，且不超过 end （不包括 end ）的序列。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.full](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.full.html#mindspore.ops.full)|New|创建一个指定shape的Tensor，并用指定值填充。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.full_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.full_like.html#mindspore.ops.full_like)|New|返回一个shape与 input 相同并且使用 fill_value 填充的Tensor。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.heaviside](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.heaviside.html#mindspore.ops.heaviside)|New|计算输入中每个元素的 Heaviside 阶跃函数。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.logspace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logspace.html#mindspore.ops.logspace)|New|返回一个大小为 steps 的1-D Tensor，其值从 $base^{start}$ 到 $base^{end}$ ，以 base 为底数。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.zeros](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.zeros.html#mindspore.ops.zeros)|New|创建一个填满0的Tensor，shape由 size 决定， dtype由 dtype 决定。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.zeros_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.zeros_like.html#mindspore.ops.zeros_like)|New|创建一个填满0的Tensor，shape由 input 决定，dtype由 dtype 决定。|r2.0: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.col2im](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.col2im.html#mindspore.ops.col2im)|Changed|将一组滑动局部块组合成一个大的Tensor。|r1.10: GPU => r2.0: Ascend/GPU/CPU|r1.10: Array操作 => r2.0: 图像函数
[mindspore.ops.pad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pad.html#mindspore.ops.pad)|Changed|r1.10: 根据参数 paddings 对输入进行填充。 => r2.0: 根据参数 padding 对输入进行填充。|r1.10: Ascend/GPU/CPU => r2.0: GPU/CPU|r1.10: 神经网络 => r2.0: 图像函数
[mindspore.ops.bernoulli](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bernoulli.html#mindspore.ops.bernoulli)|Changed|r1.10: 以p的概率随机将输出的元素设置为0或1，服从伯努利分布。 => r2.0: 以 p 的概率随机将输出的元素设置为0或1，服从伯努利分布。|r1.10: GPU => r2.0: GPU/CPU|r1.10: 逐元素运算 => r2.0: 随机生成函数
[mindspore.ops.grid_sample](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.grid_sample.html#mindspore.ops.grid_sample)|Changed|给定一个输入和一个网格，使用网格中的输入值和像素位置计算输出。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|r1.10: 采样函数 => r2.0: 图像函数
[mindspore.ops.core](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.core.html#mindspore.ops.core)|Deleted|A decorator that adds a flag to the function.|Ascend/GPU/CPU|其他函数
[mindspore.ops.affine_grid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.affine_grid.html#mindspore.ops.affine_grid)|New|基于输入的批量仿射矩阵 theta ，返回一个二维或三维的流场（采样网格）。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.bounding_box_decode](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bounding_box_decode.html#mindspore.ops.bounding_box_decode)|New|解码边界框位置信息，计算偏移量，此算子将偏移量转换为Bbox，用于在后续图像中标记目标等。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.bounding_box_encode](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bounding_box_encode.html#mindspore.ops.bounding_box_encode)|New|编码边界框位置信息，计算预测边界框和真实边界框之间的偏移，并将此偏移作为损失变量。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.check_valid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.check_valid.html#mindspore.ops.check_valid)|New|检查边界框是否在图片内。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.crop_and_resize](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.crop_and_resize.html#mindspore.ops.crop_and_resize)|New|对输入图像Tensor进行裁剪并调整其大小。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.pixel_shuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pixel_shuffle.html#mindspore.ops.pixel_shuffle)|New|对输入 input 应用像素重组操作，它实现了步长为 $1/r$ 的子像素卷积。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.pixel_unshuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pixel_unshuffle.html#mindspore.ops.pixel_unshuffle)|New|对输入 input 应用逆像素重组操作，这是像素重组的逆操作。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.upsample](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.upsample.html#mindspore.ops.upsample)|New|mindspore.ops.interpolate() 的别名。|r2.0: Ascend/GPU/CPU|图像函数
[mindspore.ops.grad](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.grad.html#mindspore.ops.grad) => [mindspore.grad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.grad.html#mindspore.grad)|Moved|生成求导函数，用于计算给定函数的梯度。|Ascend/GPU/CPU|微分函数
[mindspore.ops.jvp](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.jvp.html#mindspore.ops.jvp) => [mindspore.jvp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.jvp.html#mindspore.jvp)|Moved|计算给定网络的雅可比向量积(Jacobian-vector product, JVP)。|Ascend/GPU/CPU|微分函数
[mindspore.ops.stop_gradient](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.stop_gradient.html#mindspore.ops.stop_gradient)|New|用于消除某个值对梯度的影响，例如截断来自于函数输出的梯度传播。|r2.0: Ascend/GPU/CPU|微分函数
[mindspore.ops.value_and_grad](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.value_and_grad.html#mindspore.ops.value_and_grad) => [mindspore.value_and_grad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.value_and_grad.html#mindspore.value_and_grad)|Moved|生成求导函数，用于计算给定函数的正向计算结果和梯度。|Ascend/GPU/CPU|微分函数
[mindspore.ops.vjp](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.vjp.html#mindspore.ops.vjp) => [mindspore.vjp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.vjp.html#mindspore.vjp)|Moved|计算给定网络的向量雅可比积(vector-jacobian-product, VJP)。|Ascend/GPU/CPU|微分函数
[mindspore.ops.vmap](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.vmap.html#mindspore.ops.vmap) => [mindspore.vmap](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.vmap.html#mindspore.vmap)|Moved|自动向量化（Vectorizing Map，vmap），是一种用于沿参数轴映射函数 fn 的高阶函数。|Ascend/GPU/CPU|微分函数
[mindspore.ops.binary_cross_entropy](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.binary_cross_entropy.html#mindspore.ops.binary_cross_entropy)|New|计算预测值 logits 和 目标值 labels 之间的二值交叉熵（度量两个概率分布间的差异性信息）损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.cosine_embedding_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cosine_embedding_loss.html#mindspore.ops.cosine_embedding_loss)|New|余弦相似度损失函数，用于测量两个Tensor之间的相似性。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.ctc_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ctc_loss.html#mindspore.ops.ctc_loss)|New|计算CTC（Connectist Temporal Classification）损失和梯度。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.gaussian_nll_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.gaussian_nll_loss.html#mindspore.ops.gaussian_nll_loss)|New|服从高斯分布的负对数似然损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.hinge_embedding_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hinge_embedding_loss.html#mindspore.ops.hinge_embedding_loss)|New|Hinge Embedding 损失函数，衡量输入 inputs 和标签 targets （包含1或-1）之间的损失值。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.huber_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.huber_loss.html#mindspore.ops.huber_loss)|New|计算预测值和目标值之间的误差，兼具 mindspore.ops.l1_loss() 和 mindspore.ops.mse_loss() 的优点。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.l1_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.l1_loss.html#mindspore.ops.l1_loss)|New|用于计算预测值和目标值之间的平均绝对误差。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.margin_ranking_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.margin_ranking_loss.html#mindspore.ops.margin_ranking_loss)|New|排序损失函数，用于创建一个衡量给定损失的标准。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.mse_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mse_loss.html#mindspore.ops.mse_loss)|New|计算预测值和标签值之间的均方误差。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.multi_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multi_margin_loss.html#mindspore.ops.multi_margin_loss)|New|用于优化多分类问题的合页损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.multilabel_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multilabel_margin_loss.html#mindspore.ops.multilabel_margin_loss)|New|用于优化多标签分类问题的合页损失。|r2.0: Ascend/GPU|损失函数
[mindspore.ops.multilabel_soft_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multilabel_soft_margin_loss.html#mindspore.ops.multilabel_soft_margin_loss)|New|基于最大熵计算用于多标签优化的损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.ops.triplet_margin_loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.triplet_margin_loss.html#mindspore.ops.triplet_margin_loss)|New|三元组损失函数。|r2.0: GPU|损失函数
[mindspore.ops.argsort](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.argsort.html#mindspore.ops.argsort)|New|按指定顺序对输入Tensor沿给定维度进行排序，并返回排序后的索引。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.greater](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.greater.html#mindspore.ops.greater)|New|按元素比较输入参数 $input > other$ 的值，输出结果为bool值。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.greater_equal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.greater_equal.html#mindspore.ops.greater_equal)|New|按元素比较输入参数 $input \geq other$ 的值，输出结果为bool值。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.is_complex](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.is_complex.html#mindspore.ops.is_complex)|New|如果Tensor的数据类型是复数，则返回True，否则返回False。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.is_floating_point](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.is_floating_point.html#mindspore.ops.is_floating_point)|New|判断 input 的dtype是否是浮点数据类型，包括mindspore.float64，mindspore.float32，mindspore.float16。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isinf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isinf.html#mindspore.ops.isinf)|New|确定输入Tensor每个位置上的元素是否为无穷大或无穷小。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isneginf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isneginf.html#mindspore.ops.isneginf)|New|逐元素判断是否是负inf。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isposinf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isposinf.html#mindspore.ops.isposinf)|New|逐元素判断是否是正inf。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.isreal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.isreal.html#mindspore.ops.isreal)|New|逐元素判断是否为实数。|r2.0: GPU/CPU|比较函数
[mindspore.ops.less_equal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.less_equal.html#mindspore.ops.less_equal)|New|逐元素计算 $input <= other$ 的bool值。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.lt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lt.html#mindspore.ops.lt)|New|mindspore.ops.less() 的别名。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.msort](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.msort.html#mindspore.ops.msort)|New|将输入Tensor的元素沿其第一个维度按值升序排序。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.not_equal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.not_equal.html#mindspore.ops.not_equal)|New|mindspore.ops.ne() 的别名。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.same_type_shape](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.same_type_shape.html#mindspore.ops.same_type_shape)|Deleted|Checks whether the data type and shape of two tensors are the same.|Ascend/GPU/CPU|比较函数
[mindspore.ops.searchsorted](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.searchsorted.html#mindspore.ops.searchsorted)|New|返回位置索引，根据这个索引将 values 插入 sorted_sequence 后，sorted_sequence 的最内维度的顺序保持不变。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.topk](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.topk.html#mindspore.ops.topk)|New|沿给定维度查找 k 个最大或最小元素和对应的索引。|r2.0: Ascend/GPU/CPU|比较函数
[mindspore.ops.elu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.elu.html#mindspore.ops.elu)|New|指数线性单元激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.gelu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.gelu.html#mindspore.ops.gelu)|New|高斯误差线性单元激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.glu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.glu.html#mindspore.ops.glu)|New|门线性单元函数（Gated Linear Unit function）。|r2.0: Ascend/CPU|激活函数
[mindspore.ops.hardsigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hardsigmoid.html#mindspore.ops.hardsigmoid)|New|Hard Sigmoid激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.hardtanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hardtanh.html#mindspore.ops.hardtanh)|New|逐元素元素计算hardtanh激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.leaky_relu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.leaky_relu.html#mindspore.ops.leaky_relu)|New|leaky_relu激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.logsigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logsigmoid.html#mindspore.ops.logsigmoid)|New|按元素计算logsigmoid激活函数。|r2.0: Ascend/GPU|激活函数
[mindspore.ops.prelu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.prelu.html#mindspore.ops.prelu)|New|带参数的线性修正单元激活函数（Parametric Rectified Linear Unit activation function）。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.relu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.relu.html#mindspore.ops.relu)|New|对输入Tensor逐元素计算线性修正单元激活函数（Rectified Linear Unit）值。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.relu6](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.relu6.html#mindspore.ops.relu6)|New|计算输入Tensor的ReLU（修正线性单元），其上限为6。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.rrelu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rrelu.html#mindspore.ops.rrelu)|New|Randomized Leaky ReLU激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.silu](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.silu.html#mindspore.ops.silu)|New|按输入逐元素计算激活函数SiLU（Sigmoid Linear Unit）。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.soft_shrink](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.soft_shrink.html#mindspore.ops.soft_shrink) => [mindspore.ops.softshrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.softshrink.html#mindspore.ops.softshrink)|Changed|Soft Shrink激活函数，按输入元素计算输出。|Ascend/CPU/GPU|激活函数
[mindspore.ops.softmin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.softmin.html#mindspore.ops.softmin)|New|在指定轴上对输入Tensor执行Softmin函数做归一化操作。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.softshrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.softshrink.html#mindspore.ops.softshrink)|New|逐元素计算Soft Shrink激活函数。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.threshold](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.threshold.html#mindspore.ops.threshold)|New|使用阈值 thr 参数对 input 逐元素阈值化，并将其结果作为Tensor返回。|r2.0: Ascend/GPU/CPU|激活函数
[mindspore.ops.adaptive_avg_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adaptive_avg_pool1d.html#mindspore.ops.adaptive_avg_pool1d)|New|对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应平均池化操作。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.adaptive_avg_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adaptive_avg_pool3d.html#mindspore.ops.adaptive_avg_pool3d)|Changed|r1.10: 对由多个平面组成的输入Tensor，进行三维的自适应平均池化操作。 => r2.0: 对一个多平面输入信号执行三维自适应平均池化。|r1.10: GPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.adaptive_max_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adaptive_max_pool1d.html#mindspore.ops.adaptive_max_pool1d)|New|对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应最大池化操作。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.avg_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.avg_pool1d.html#mindspore.ops.avg_pool1d)|New|在输入Tensor上应用1D平均池化，输入Tensor可以看作是由一系列1D平面组成的。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.avg_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.avg_pool3d.html#mindspore.ops.avg_pool3d)|New|在输入Tensor上应用3D平均池化，输入Tensor可以看作是由一系列3D平面组成的。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.batch_norm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.batch_norm.html#mindspore.ops.batch_norm)|New|对输入数据进行批量归一化和更新参数。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.conv1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.conv1d.html#mindspore.ops.conv1d)|New|对输入Tensor计算一维卷积。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.conv3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.conv3d.html#mindspore.ops.conv3d)|New|对输入Tensor计算三维卷积。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.ctc_greedy_decoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ctc_greedy_decoder.html#mindspore.ops.ctc_greedy_decoder)|Changed|对输入中给定的logits执行贪婪解码。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.dropout1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dropout1d.html#mindspore.ops.dropout1d)|New|在训练期间，以服从伯努利分布的概率 p 随机将输入Tensor的某些通道归零（对于形状为 $NCL$ 的三维Tensor，其通道特征图指的是后一维 $L$ 的一维特征图）。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.fold](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fold.html#mindspore.ops.fold)|New|将提取出的滑动局部区域块还原成更大的输出Tensor。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.fractional_max_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fractional_max_pool3d.html#mindspore.ops.fractional_max_pool3d)|New|在输入 input 上应用三维分数最大池化。|r2.0: GPU/CPU|神经网络
[mindspore.ops.lp_pool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lp_pool1d.html#mindspore.ops.lp_pool1d)|New|在输入Tensor上应用1D LP池化运算，可被视为组成一个1D输入平面。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.lp_pool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lp_pool2d.html#mindspore.ops.lp_pool2d)|New|在输入Tensor上应用2D LP池化运算，可被视为组成一个2D输入平面。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_pool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_pool2d.html#mindspore.ops.max_pool2d)|New|二维最大值池化。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_pool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_pool3d.html#mindspore.ops.max_pool3d)|Changed|三维最大值池化。|r1.10: GPU => r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_unpool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_unpool1d.html#mindspore.ops.max_unpool1d)|New|max_pool1d 的逆过程。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_unpool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_unpool2d.html#mindspore.ops.max_unpool2d)|New|max_pool2d 的逆过程。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.max_unpool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.max_unpool3d.html#mindspore.ops.max_unpool3d)|New|mindspore.ops.max_pool3d() 的逆过程。|r2.0: Ascend/GPU/CPU|神经网络
[mindspore.ops.is_tensor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.is_tensor.html#mindspore.ops.is_tensor)|New|判断输入对象是否为 mindspore.Tensor。|r2.0: Ascend/GPU/CPU|类型转换
[mindspore.ops.scalar_to_array](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.scalar_to_array.html#mindspore.ops.scalar_to_array)|Deleted|将Scalar转换为 Tensor 。|Ascend/GPU/CPU|类型转换
[mindspore.ops.addbmm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.addbmm.html#mindspore.ops.addbmm)|New|对 batch1 和 batch2 应用批量矩阵乘法后进行reduced add， input 和最终的结果相加。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.addmm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.addmm.html#mindspore.ops.addmm)|New|对 mat1 和 mat2 应用矩阵乘法。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.addr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.addr.html#mindspore.ops.addr)|New|计算 vec1 和 vec2 的外积，并将其添加到 x 中。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.adjoint](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.adjoint.html#mindspore.ops.adjoint)|New|逐元素计算Tensor的共轭，并转置最后两个维度。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.baddbmm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.baddbmm.html#mindspore.ops.baddbmm)|New|对输入的两个三维矩阵batch1与batch2相乘，并将结果与input相加。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.bmm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bmm.html#mindspore.ops.bmm)|New|基于batch维度的两个Tensor的矩阵乘法。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.cholesky](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cholesky.html#mindspore.ops.cholesky)|New|计算对称正定矩阵或一批对称正定矩阵的Cholesky分解。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.eig](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.eig.html#mindspore.ops.eig)|New|计算输入方阵（batch方阵）的特征值和特征向量。|r2.0: Ascend/CPU|线性代数函数
[mindspore.ops.geqrf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.geqrf.html#mindspore.ops.geqrf)|New|将矩阵分解为正交矩阵 Q 和上三角矩阵 R 的乘积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.inner](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inner.html#mindspore.ops.inner)|New|计算两个1D Tensor的点积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.inverse](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.inverse.html#mindspore.ops.inverse)|New|计算输入矩阵的逆。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.kron](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.kron.html#mindspore.ops.kron)|New|计算 x 和 y 的Kronecker积：$x ⊗ y$ 。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.logdet](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logdet.html#mindspore.ops.logdet)|New|计算方块矩阵或批量方块矩阵的对数行列式。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.lu_unpack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lu_unpack.html#mindspore.ops.lu_unpack)|New|将 LU_data 和 LU_pivots 还原为P, L, U矩阵，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.matrix_solve](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.matrix_solve.html#mindspore.ops.matrix_solve)|Changed|求解线性方程组。|r1.10: GPU/CPU => r2.0: Ascend/CPU|线性代数函数
[mindspore.ops.mm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mm.html#mindspore.ops.mm)|New|计算两个矩阵的乘积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.mv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mv.html#mindspore.ops.mv)|New|实现矩阵 mat 和向量 vec 相乘。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.orgqr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.orgqr.html#mindspore.ops.orgqr)|New|计算 mindspore.ops.Geqrf 返回的正交矩阵 $Q$ 的显式表示。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.outer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.outer.html#mindspore.ops.outer)|New|计算 input 和 vec2 的外积。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.pinv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pinv.html#mindspore.ops.pinv)|New|计算矩阵的（Moore-Penrose）伪逆。|r2.0: CPU|线性代数函数
[mindspore.ops.qr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.qr.html#mindspore.ops.qr)|New|返回一个或多个矩阵的QR（正交三角）分解。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.slogdet](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.slogdet.html#mindspore.ops.slogdet)|New|对一个或多个方阵行列式的绝对值取对数，返回其符号和值。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.svd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd)|New|计算单个或多个矩阵的奇异值分解。|r2.0: GPU/CPU|线性代数函数
[mindspore.ops.trace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.trace.html#mindspore.ops.trace)|New|返回input的对角线方向上的总和。|r2.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.print_](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.print_.html#mindspore.ops.print_)|New|将输入数据进行打印输出。|r2.0: Ascend/GPU/CPU|调试函数
[mindspore.ops.bartlett_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bartlett_window.html#mindspore.ops.bartlett_window)|New|Bartlett窗口函数是一种三角形状的加权函数，通常用于平滑处理或频域分析信号。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.blackman_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.blackman_window.html#mindspore.ops.blackman_window)|New|布莱克曼窗口函数，常用来为FFT截取有限长的信号片段。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.hamming_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hamming_window.html#mindspore.ops.hamming_window)|New|返回一个Hamming window。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.hann_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hann_window.html#mindspore.ops.hann_window)|New|生成一个Hann window。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.kaiser_window](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.kaiser_window.html#mindspore.ops.kaiser_window)|New|生成一个Kaiser window，也叫做Kaiser-Bessel window。|r2.0: Ascend/GPU/CPU|谱函数
[mindspore.ops.cdist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cdist.html#mindspore.ops.cdist)|Changed|r1.10: 批量计算两个Tensor每一批次所有向量两两之间的p-范数距离。 => r2.0: 计算两个Tensor每对列向量之间的p-norm距离。|r1.10: Ascend/CPU => r2.0: Ascend/GPU/CPU|距离函数
[mindspore.ops.dist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.dist.html#mindspore.ops.dist)|New|计算输入中每对行向量之间的 $p$-norm距离。|r2.0: Ascend/GPU/CPU|距离函数
[mindspore.ops.pdist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.pdist.html#mindspore.ops.pdist)|New|计算输入中每对行向量之间的p-范数距离。|r2.0: GPU/CPU|距离函数
[mindspore.ops.absolute](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.absolute.html#mindspore.ops.absolute)|New|mindspore.ops.abs() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.accumulate_n](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.accumulate_n.html#mindspore.ops.accumulate_n)|New|逐元素将所有输入的Tensor相加。|r2.0: Ascend|逐元素运算
[mindspore.ops.addcmul](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.addcmul.html#mindspore.ops.addcmul)|Changed|r1.10: Performs the element-wise product of tensor x1 and tensor x2, multiply the result by the scalar value and add it to input_data. => r2.0: 执行Tensor tensor1 与Tensor tensor2 的逐元素乘积，将结果乘以标量值 value ，并将其添加到 input 中。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.addmv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.addmv.html#mindspore.ops.addmv)|New|mat 和 vec 相乘，且将输入向量 x 加到最终结果中。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.angle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.angle.html#mindspore.ops.angle)|New|逐元素计算复数Tensor的辐角。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arccos](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arccos.html#mindspore.ops.arccos)|New|mindspore.ops.acos() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arccosh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arccosh.html#mindspore.ops.arccosh)|New|详情请参考 mindspore.ops.acosh()。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arcsin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arcsin.html#mindspore.ops.arcsin)|New|mindspore.ops.asin() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arcsinh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arcsinh.html#mindspore.ops.arcsinh)|New|mindspore.ops.asinh() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arctan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arctan.html#mindspore.ops.arctan)|New|详情请参考 mindspore.ops.atan()。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arctan2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arctan2.html#mindspore.ops.arctan2)|New|详情请参考 mindspore.ops.atan2()。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.arctanh](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.arctanh.html#mindspore.ops.arctanh)|New|mindspore.ops.atanh() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.atleast_1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.atleast_1d.html#mindspore.ops.atleast_1d)|New|调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于1。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.atleast_2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.atleast_2d.html#mindspore.ops.atleast_2d)|New|调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于2。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.atleast_3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.atleast_3d.html#mindspore.ops.atleast_3d)|New|调整 inputs 中的Tensor维度，使输入中每个Tensor维度不低于3。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.bitwise_left_shift](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bitwise_left_shift.html#mindspore.ops.bitwise_left_shift)|New|逐元素对输入 input 进行左移位运算, 移动的位数由 other 指定。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.bitwise_right_shift](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.bitwise_right_shift.html#mindspore.ops.bitwise_right_shift)|New|逐元素对输入 input 进行右移位运算, 移动的位数由 other 指定。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.clamp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.clamp.html#mindspore.ops.clamp)|New|将输入Tensor的值裁剪到指定的最小值和最大值之间。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.clip](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.clip.html#mindspore.ops.clip)|New|mindspore.ops.clamp() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.combinations](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.combinations.html#mindspore.ops.combinations)|New|返回输入Tensor中元素的所有长度为 r 的子序列。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.copysign](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.copysign.html#mindspore.ops.copysign)|New|逐元素地创建一个新的浮点Tensor，其大小为 x，符号为 other 的符号。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.cosine_similarity](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cosine_similarity.html#mindspore.ops.cosine_similarity)|New|沿轴计算的x1和x2之间的余弦相似度。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.cov](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cov.html#mindspore.ops.cov)|New|给定输入 input 和权重，返回输入 input 的协方差矩阵(每对变量的协方差的方阵)，其中输入行是变量，列是观察值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.deg2rad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.deg2rad.html#mindspore.ops.deg2rad)|New|逐元素地将 x 从度数制转换为弧度制。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.diag_embed](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diag_embed.html#mindspore.ops.diag_embed)|New|生成一个Tensor，其对角线值由 input 中的值填充，其余位置置0。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.diff](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.diff.html#mindspore.ops.diff)|New|沿着给定维度计算输入Tensor x 的n阶前向差分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.digamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.digamma.html#mindspore.ops.digamma)|New|计算lgamma对数函数在输入上的梯度。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.divide](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.divide.html#mindspore.ops.divide)|New|mindspore.ops.div() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.erfinv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.erfinv.html#mindspore.ops.erfinv)|New|计算输入的逆误差函数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.exp2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.exp2.html#mindspore.ops.exp2)|New|逐元素计算Tensor input 以2为底的指数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.float_power](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power)|New|计算 input 的指数幂。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.fmod](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.fmod.html#mindspore.ops.fmod)|New|计算除法运算 input/other 的浮点余数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.frac](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.frac.html#mindspore.ops.frac)|New|计算 x 中每个元素的小数部分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.gcd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.gcd.html#mindspore.ops.gcd)|New|按元素计算输入Tensor的最大公约数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.hypot](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.hypot.html#mindspore.ops.hypot)|New|按元素计算以输入Tensor为直角边的三角形的斜边。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.i0](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.i0.html#mindspore.ops.i0)|New|mindspore.ops.bessel_i0() 的别名。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.igamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.igamma.html#mindspore.ops.igamma)|New|计算正则化的下层不完全伽马函数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.igammac](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.igammac.html#mindspore.ops.igammac)|New|计算正则化的上层不完全伽马函数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.imag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.imag.html#mindspore.ops.imag)|New|返回包含输入Tensor的虚部。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.lcm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.lcm.html#mindspore.ops.lcm)|New|逐元素计算两个输入Tensor的最小公倍数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.ldexp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ldexp.html#mindspore.ops.ldexp)|New|逐元素将输入Tensor乘以 $2^{other}$ 。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.log10](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.log10.html#mindspore.ops.log10)|New|逐元素返回Tensor以10为底的对数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.log2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.log2.html#mindspore.ops.log2)|New|逐元素返回Tensor以2为底的对数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.logaddexp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logaddexp.html#mindspore.ops.logaddexp)|New|计算输入的指数和的对数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.logaddexp2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logaddexp2.html#mindspore.ops.logaddexp2)|New|计算以2为底的输入的指数和的对数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.logical_xor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logical_xor.html#mindspore.ops.logical_xor)|New|逐元素计算两个Tensor的逻辑异或运算。|r2.0: Ascend/CPU|逐元素运算
[mindspore.ops.logit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.logit.html#mindspore.ops.logit)|New|逐元素计算Tensor的logit值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.multiply](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multiply.html#mindspore.ops.multiply)|New|mindspore.ops.mul() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.mvlgamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.mvlgamma.html#mindspore.ops.mvlgamma)|New|逐元素计算 p 维多元对数伽马函数值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.negative](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.negative.html#mindspore.ops.negative)|New|mindspore.ops.neg() 的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.nextafter](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.nextafter.html#mindspore.ops.nextafter)|New|逐元素返回 input 指向 other 的下一个可表示值符点值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.polar](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.polar.html#mindspore.ops.polar)|New|将极坐标转化为笛卡尔坐标。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.polygamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.polygamma.html#mindspore.ops.polygamma)|New|计算关于 input 的多伽马函数的 $n$ 阶导数。|r2.0: GPU/CPU|逐元素运算
[mindspore.ops.positive](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.positive.html#mindspore.ops.positive)|New|返回输入Tensor。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.rad2deg](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rad2deg.html#mindspore.ops.rad2deg)|New|逐元素地将 x 从弧度制转换为度数制。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.ravel](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ravel.html#mindspore.ops.ravel)|New|沿着0轴方向，将多维Tensor展开成一维。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.real](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.real.html#mindspore.ops.real)|New|返回输入Tensor的实数部分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.reciprocal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.reciprocal.html#mindspore.ops.reciprocal)|New|返回输入Tensor每个元素的倒数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.remainder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.remainder.html#mindspore.ops.remainder)|New|逐元素计算第一个元素除第二个元素的余数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.rot90](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rot90.html#mindspore.ops.rot90)|New|沿轴指定的平面内将n-D Tensor旋转90度。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.rsqrt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rsqrt.html#mindspore.ops.rsqrt)|New|逐元素计算输入Tensor元素的平方根倒数。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sgn](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sgn.html#mindspore.ops.sgn)|New|mindspore.ops.sign() 在复数上的扩展。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sign](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sign.html#mindspore.ops.sign)|New|按sign公式逐元素计算输入Tensor。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.signbit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.signbit.html#mindspore.ops.signbit)|New|判断每个元素的符号，如果元素值小于0则对应输出的位置为True，否则为False。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sinc](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sinc.html#mindspore.ops.sinc)|New|计算输入的归一化正弦值。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.sqrt](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.sqrt.html#mindspore.ops.sqrt)|New|逐元素返回当前Tensor的平方根。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.subtract](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.subtract.html#mindspore.ops.subtract)|New|对Tensor进行逐元素的减法。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.t](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.t.html#mindspore.ops.t)|New|转置二维Tensor。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.tanhshrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tanhshrink.html#mindspore.ops.tanhshrink)|New|Tanhshrink激活函数， $Tanhshrink(x)=x-Tanh(x)$ ，其中 $x$ 即输入 input。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.trapz](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.trapz.html#mindspore.ops.trapz)|New|使用梯形法则沿给定轴 dim 对 y(x) 进行积分。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.tril_indices](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.tril_indices.html#mindspore.ops.tril_indices)|New|计算 row * col 行列矩阵的下三角元素的索引，并将它们作为一个 2xN 的Tensor返回。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.triu_indices](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.triu_indices.html#mindspore.ops.triu_indices)|New|计算 row * col 行列矩阵的上三角元素的索引，并将它们作为一个 2xN 的Tensor返回。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.true_divide](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.true_divide.html#mindspore.ops.true_divide)|New|mindspore.ops.div() 在 $rounding\\_mode=None$ 时的别名。|r2.0: Ascend/GPU/CPU|逐元素运算
[mindspore.ops.choice_with_mask](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.choice_with_mask.html#mindspore.ops.choice_with_mask)|New|对输入进行随机取样，返回取样索引和掩码。|r2.0: Ascend/GPU/CPU|采样函数
[mindspore.ops.log_uniform_candidate_sampler](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.log_uniform_candidate_sampler.html#mindspore.ops.log_uniform_candidate_sampler)|New|使用log-uniform(Zipfian)分布对一组类别进行采样。|r2.0: Ascend/CPU|采样函数
[mindspore.ops.laplace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.laplace.html#mindspore.ops.laplace)|Changed|r1.10: Generates random numbers according to the Laplace random number distribution. => r2.0: 根据拉普拉斯分布生成随机数。|r1.10: Ascend => r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.multinomial](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multinomial.html#mindspore.ops.multinomial)|Changed|r1.10: Returns a tensor sampled from the multinomial probability distribution located in the corresponding row of the input tensor. => r2.0: 根据输入生成一个多项式分布的Tensor。|r1.10: GPU => r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.multinomial_with_replacement](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.multinomial_with_replacement.html#mindspore.ops.multinomial_with_replacement)|New|返回一个Tensor，其中每行包含从重复采样的多项式分布中抽取的 numsamples 个索引。|r2.0: CPU|随机生成函数
[mindspore.ops.poisson](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.poisson.html#mindspore.ops.poisson)|Deleted|The ops.poisson is deprecated, please use mindspore.ops.random_poisson Generates random numbers according to the Poisson random number distribution.||随机生成函数
[mindspore.ops.rand](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rand.html#mindspore.ops.rand)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从均匀分布的 $[0, 1)$ 区间的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.rand_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.rand_like.html#mindspore.ops.rand_like)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从均匀分布的 $[0, 1)$ 区间的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randint](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randint.html#mindspore.ops.randint)|New|返回一个Tensor，其元素为 [ low , high ) 区间的随机整数。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randint_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randint_like.html#mindspore.ops.randint_like)|New|返回一个Tensor，其元素为 [ low , high ) 区间的随机整数，根据 input 决定shape和dtype。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randn](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randn.html#mindspore.ops.randn)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.randn_like](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randn_like.html#mindspore.ops.randn_like)|New|返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的数字。|r2.0: Ascend/GPU/CPU|随机生成函数
[mindspore.ops.random_poisson](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.random_poisson.html#mindspore.ops.random_poisson)|Changed|r1.10: 从各指定均值的泊松分布中，随机采样 shape 形状的随机数。 => r2.0: 从一个指定均值为 rate 的泊松分布中，随机生成形状为 shape 的随机数Tensor。|r1.10: CPU => r2.0: GPU/CPU|随机生成函数
[mindspore.ops.randperm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randperm.html#mindspore.ops.randperm)|New|生成从 0 到 n-1 的整数随机排列。|r2.0: CPU|随机生成函数
[mindspore.ops.uniform](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.uniform.html#mindspore.ops.uniform)|Changed|生成服从均匀分布的随机数。|r1.10: Ascend/GPU/CPU => r2.0: GPU/CPU|随机生成函数
