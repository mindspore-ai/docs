# mindspore.ops.primitive API接口变更

与上一版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
|[mindspore.ops.AffineGrid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AffineGrid.html#mindspore.ops.AffineGrid)|New|基于一批仿射矩阵 theta 生成一个2D 或 3D 的流场（采样网格）。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.ChannelShuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ChannelShuffle.html#mindspore.ops.ChannelShuffle)|New|将shape为 \((*, C, H, W)\) 的Tensor的通道划分成 \(g\) 组，并按如下方式重新排列 \((*, \frac C g, g, H*W)\) ，。|r2.0: Ascend/CPU|Array操作|
|[mindspore.ops.Col2Im](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Col2Im.html#mindspore.ops.Col2Im)|New|将一组通过滑窗获得的数组组合成一个大的Tensor。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Cummax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cummax.html#mindspore.ops.Cummax)|New|返回输入Tensor在指定轴上的累计最大值与其对应的索引。|r2.0: GPU/CPU|Array操作|
|[mindspore.ops.Cummin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cummin.html#mindspore.ops.Cummin)|New|返回输入Tensor在指定轴上的累计最小值与其对应的索引。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Diag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Diag.html#mindspore.ops.Diag)|New|用给定的对角线值构造对角线Tensor。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.FillDiagonal](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.FillDiagonal.html#mindspore.ops.FillDiagonal)|Deleted|填充至少具有二维的Tensor的主对角线。|GPU/CPU|Array操作|
|[mindspore.ops.FillV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.FillV2.html#mindspore.ops.FillV2)|New|创建一个Tensor，其shape由 shape 指定，其值则由 value 进行填充。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Fmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Fmax.html#mindspore.ops.Fmax)|New|逐元素计算输入Tensor的最大值。|r2.0: CPU|Array操作|
|[mindspore.ops.HammingWindow](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.HammingWindow.html#mindspore.ops.HammingWindow)|Changed|使用输入窗口长度计算汉明窗口函数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Heaviside](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Heaviside.html#mindspore.ops.Heaviside)|New|计算输入中每个元素的Heaviside步长函数。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Histogram](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Histogram.html#mindspore.ops.Histogram)|Deleted|计算Tensor的直方图。|CPU|Array操作|
|[mindspore.ops.Hypot](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Hypot.html#mindspore.ops.Hypot)|New|将输入Tensor的逐个元素作为直角三角形的直角边，并计算其斜边的值。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Igamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Igamma.html#mindspore.ops.Igamma)|New|计算正则化的下层不完全伽马函数。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.IndexFill](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.IndexFill.html#mindspore.ops.IndexFill)|New|按 index 中给定的顺序选择索引，将输入Tensor x 的 dim 维度下的元素用 value 的值填充。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.InplaceAdd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceAdd.html#mindspore.ops.InplaceAdd)|Changed|将 input_v 添加到 x 的特定行。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.InplaceIndexAdd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceIndexAdd.html#mindspore.ops.InplaceIndexAdd)|New|逐元素将一个Tensor updates 添加到原Tensor var 的指定轴和索引处。|r2.0: Ascend/CPU|Array操作|
|[mindspore.ops.InplaceSub](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceSub.html#mindspore.ops.InplaceSub)|Changed|从 x 的特定行减去 input_v 。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.InplaceUpdate](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceUpdate.html#mindspore.ops.InplaceUpdate)|Changed|r2.0.0-alpha: 将 x 的特定行更新为 v 。 => r2.0: InplaceUpdate接口已废弃，请使用接口 mindspore.ops.InplaceUpdateV2 替换，废弃版本2.0。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: |Array操作|
|[mindspore.ops.InplaceUpdateV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceUpdateV2.html#mindspore.ops.InplaceUpdateV2)|New|根据 indices，将 x 中的某些值更新为 v。|r2.0: GPU/CPU|Array操作|
|[mindspore.ops.IsClose](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.IsClose.html#mindspore.ops.IsClose)|New|返回一个bool型Tensor，表示 x1 的每个元素与 x2 的每个元素在给定容忍度内是否“接近”。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Lcm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Lcm.html#mindspore.ops.Lcm)|Changed|逐个元素计算输入Tensor的最小公倍数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.LeftShift](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LeftShift.html#mindspore.ops.LeftShift)|Changed|r2.0.0-alpha: 将Tensor每个位置的值向左移动几个比特位。 => r2.0: 将Tensor每个位置的值向左移动若干个比特位。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.ListDiff](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.ListDiff.html#mindspore.ops.ListDiff)|Deleted|比较两个数字列表之间的不同。|GPU/CPU|Array操作|
|[mindspore.ops.LogSpace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LogSpace.html#mindspore.ops.LogSpace)|Changed|返回一个大小为 steps 的1-D Tensor，其值从 \(base^{start}\) 到 \(base^{end}\) ，以 base 为底数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Lstsq](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Lstsq.html#mindspore.ops.Lstsq)|Deleted|计算满秩矩阵 x \((m \times n)\) 与满秩矩阵 a \((m \times k)\) 的最小二乘问题或最小范数问题的解。|CPU|Array操作|
|[mindspore.ops.LuUnpack](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LuUnpack.html#mindspore.ops.LuUnpack)|New|将 LU_data 和 LU_pivots 还原为为P, L, U矩阵，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。|r2.0: GPU/CPU|Array操作|
|[mindspore.ops.MatrixExp](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.MatrixExp.html#mindspore.ops.MatrixExp)|Deleted|计算方阵的矩阵指数。|CPU|Array操作|
|[mindspore.ops.MatrixPower](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.MatrixPower.html#mindspore.ops.MatrixPower)|Deleted|计算一个batch的方阵的n次幂。|Ascend/CPU|Array操作|
|[mindspore.ops.MatrixSetDiagV3](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MatrixSetDiagV3.html#mindspore.ops.MatrixSetDiagV3)|New|更新批处理矩阵对角线的值。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.MatrixSolve](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MatrixSolve.html#mindspore.ops.MatrixSolve)|New|求解线性方程组。|r2.0: Ascend/CPU|Array操作|
|[mindspore.ops.Mvlgamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Mvlgamma.html#mindspore.ops.Mvlgamma)|New|逐元素计算 p 维多元对数伽马函数值。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.NanToNum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.NanToNum.html#mindspore.ops.NanToNum)|New|将输入中的 NaN 、正无穷大和负无穷大值分别替换为 nan 、 posinf 和 neginf 指定的值。|r2.0: Ascend/CPU|Array操作|
|[mindspore.ops.NonZero](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.NonZero.html#mindspore.ops.NonZero)|New|计算输入Tensor中所有非零元素的索引位置。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Qr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Qr.html#mindspore.ops.Qr)|New|返回一个或多个矩阵的QR（正交三角）分解。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.RandomShuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.RandomShuffle.html#mindspore.ops.RandomShuffle)|New|沿着Tensor的第一个维度进行随机打乱操作。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Range](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Range.html#mindspore.ops.Range)|Changed|返回从 start 开始，步长为 delta ，且不超过 limit （不包括 limit ）的序列。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: GPU/CPU|Array操作|
|[mindspore.ops.Renorm](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Renorm.html#mindspore.ops.Renorm)|New|对Tensor沿着指定维度 dim 进行重新规范化，要求每个子Tensor的 p 范数不超过 maxnorm 。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.ResizeNearestNeighborV2](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.ResizeNearestNeighborV2.html#mindspore.ops.ResizeNearestNeighborV2)|Deleted|使用最近邻算法将输入Tensor调整为特定大小。|CPU|Array操作|
|[mindspore.ops.Roll](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Roll.html#mindspore.ops.Roll)|Deleted|沿轴移动Tensor的元素。|Ascend/GPU|Array操作|
|[mindspore.ops.STFT](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.STFT.html#mindspore.ops.STFT)|Deleted|通过STFT量化非平稳信号频率和相位随时间的变化。|Ascend/GPU/CPU|Array操作|
|[mindspore.ops.ScatterAddWithAxis](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.ScatterAddWithAxis.html#mindspore.ops.ScatterAddWithAxis)|Deleted|该操作的输出是通过创建输入 input_x 的副本，然后将 updates 指定的值添加到 indices 指定的位置来更新副本中的值。|CPU|Array操作|
|[mindspore.ops.ScatterNdMin](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterNdMin.html#mindspore.ops.ScatterNdMin)|New|对张量中的单个值或切片应用sparse minimum。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.SearchSorted](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SearchSorted.html#mindspore.ops.SearchSorted)|New|返回位置索引，根据这个索引将 values 插入 sorted_sequence 后，sorted_sequence 的元素大小顺序保持不变。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.Tril](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Tril.html#mindspore.ops.Tril)|New|返回单个或一批二维矩阵下三角形部分，其他位置的元素将被置零。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.TrilIndices](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TrilIndices.html#mindspore.ops.TrilIndices)|New|计算 row * col 行列矩阵的下三角元素的索引，并将它们作为一个 2xN 的Tensor返回。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.TriuIndices](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TriuIndices.html#mindspore.ops.TriuIndices)|New|返回一个包含 row * col 的矩阵的上三角形部分的索引的Tensor。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.UniqueConsecutive](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.UniqueConsecutive.html#mindspore.ops.UniqueConsecutive)|New|对输入张量中连续且重复的元素去重。|r2.0: Ascend/GPU/CPU|Array操作|
|[mindspore.ops.EuclideanNorm](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.EuclideanNorm.html#mindspore.ops.EuclideanNorm)|Deleted|计算Tensor维度上元素的欧几里得范数，根据给定的轴对输入进行规约操作。|GPU|Reduction算子|
|[mindspore.ops.Median](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Median.html#mindspore.ops.Median)|New|输出Tensor指定维度 axis 上的中值与其对应的索引。|r2.0: GPU/CPU|Reduction算子|
|[mindspore.ops.ReduceStd](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.ReduceStd.html#mindspore.ops.ReduceStd)|Deleted|返回输入Tensor在 axis 维上每一行的标准差和均值。|Ascend/CPU|Reduction算子|
|[mindspore.ops.Fill](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Fill.html#mindspore.ops.Fill)|Changed|r2.0.0-alpha: 创建一个指定shape的Tensor，并用指定值填充。 => r2.0: Fill接口已弃用， 请使用 mindspore.ops.FillV2 。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: |Tensor创建|
|[mindspore.ops.Zeros](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Zeros.html#mindspore.ops.Zeros)|Changed|创建一个值全为0的Tensor。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: |Tensor创建|
|[mindspore.ops.AdaptiveAvgPool3D](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AdaptiveAvgPool3D.html#mindspore.ops.AdaptiveAvgPool3D)|New|三维自适应平均池化。|r2.0: Ascend/GPU/CPU|优化器|
|[mindspore.ops.AdaptiveMaxPool3D](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.AdaptiveMaxPool3D.html#mindspore.ops.AdaptiveMaxPool3D)|Deleted|三维自适应最大值池化。|GPU/CPU|优化器|
|[mindspore.ops.SparseApplyFtrlV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SparseApplyFtrlV2.html#mindspore.ops.SparseApplyFtrlV2)|Changed|r2.0.0-alpha: 根据FTRL-proximal算法更新相关参数。 => r2.0: ops.SparseApplyFtrlV2 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 ops.SparseApplyFtrl 代替。|r2.0.0-alpha: Ascend => r2.0: |优化器|
|[mindspore.ops.SparseApplyProximalAdagrad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SparseApplyProximalAdagrad.html#mindspore.ops.SparseApplyProximalAdagrad)|Changed|根据Proximal Adagrad算法更新网络参数。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: Ascend/GPU|优化器|
|[mindspore.ops.BartlettWindow](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BartlettWindow.html#mindspore.ops.BartlettWindow)|Changed|巴特利特窗口函数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|光谱算子|
|[mindspore.ops.BlackmanWindow](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BlackmanWindow.html#mindspore.ops.BlackmanWindow)|Changed|布莱克曼窗口函数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|光谱算子|
|[mindspore.ops.AdjustHue](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.AdjustHue.html#mindspore.ops.AdjustHue)|Deleted|调整 RGB 图像的色调。|GPU/CPU|图像处理|
|[mindspore.ops.CombinedNonMaxSuppression](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.CombinedNonMaxSuppression.html#mindspore.ops.CombinedNonMaxSuppression)|Deleted|根据分数降序，使用非极大抑制法通过遍历所有可能的边界框（Bounding Box）来选择一个最优的结果。|Ascend/CPU|图像处理|
|[mindspore.ops.ExtractGlimpse](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.ExtractGlimpse.html#mindspore.ops.ExtractGlimpse)|Deleted|从输入图像Tensor中提取glimpse，并返回一个窗口。|GPU/CPU|图像处理|
|[mindspore.ops.HSVToRGB](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.HSVToRGB.html#mindspore.ops.HSVToRGB)|Deleted|将一个或多个图像从HSV转换为RGB。|GPU/CPU|图像处理|
|[mindspore.ops.RGBToHSV](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.RGBToHSV.html#mindspore.ops.RGBToHSV)|Deleted|将一张或多张图片由RGB格式转换为HSV格式。|GPU/CPU|图像处理|
|[mindspore.ops.ResizeBilinearV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ResizeBilinearV2.html#mindspore.ops.ResizeBilinearV2)|New|使用双线性插值调整图像大小到指定的大小。|r2.0: Ascend/GPU/CPU|图像处理|
|[mindspore.ops.SampleDistortedBoundingBoxV2](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.SampleDistortedBoundingBoxV2.html#mindspore.ops.SampleDistortedBoundingBoxV2)|Deleted|为图像生成单个随机扭曲的边界框。|CPU|图像处理|
|[mindspore.ops.ScaleAndTranslate](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.ScaleAndTranslate.html#mindspore.ops.ScaleAndTranslate)|Deleted|根缩放并平移输入图像Tensor。|GPU/CPU|图像处理|
|[mindspore.ops.CTCLossV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.CTCLossV2.html#mindspore.ops.CTCLossV2)|New|计算CTC(Connectionist Temporal Classification)损失和梯度。|r2.0: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.MultiMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MultiMarginLoss.html#mindspore.ops.MultiMarginLoss)|New|创建一个损失函数，用于优化输入和输出之间的多分类合页损失。|r2.0: Ascend/GPU/CPU|损失函数|
|[mindspore.ops.TripletMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TripletMarginLoss.html#mindspore.ops.TripletMarginLoss)|New|三元组损失函数。|r2.0: GPU|损失函数|
|[mindspore.ops.Betainc](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Betainc.html#mindspore.ops.Betainc)|Deleted|计算正则化不完全beta积分 \(I_{x}(a, b)\)。|Ascend/GPU/CPU|数学运算算子|
|[mindspore.ops.Bucketize](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Bucketize.html#mindspore.ops.Bucketize)|Deleted|根据 boundaries 对 input 进行桶化。|GPU/CPU|数学运算算子|
|[mindspore.ops.Cholesky](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cholesky.html#mindspore.ops.Cholesky)|New|计算单个或成批对称正定矩阵的Cholesky分解。|r2.0: GPU/CPU|数学运算算子|
|[mindspore.ops.CompareAndBitpack](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.CompareAndBitpack.html#mindspore.ops.CompareAndBitpack)|Deleted|比较 x 和 threshold 的值（ x > threshold ），并将比较结果作为二进制数转换为uint8格式。|Ascend/GPU/CPU|数学运算算子|
|[mindspore.ops.Complex](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Complex.html#mindspore.ops.Complex)|Changed|给定复数的实部与虚部，返回一个复数的Tensor。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|数学运算算子|
|[mindspore.ops.ComplexAbs](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ComplexAbs.html#mindspore.ops.ComplexAbs)|New|返回输入复数的模。|r2.0: Ascend/GPU/CPU|数学运算算子|
|[mindspore.ops.Cross](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cross.html#mindspore.ops.Cross)|New|返回 x1 和 x2 沿着维度 dim 上的向量积（叉积）。|r2.0: Ascend/CPU|数学运算算子|
|[mindspore.ops.Gcd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Gcd.html#mindspore.ops.Gcd)|Changed|逐元素计算输入Tensor的最大公约数。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|数学运算算子|
|[mindspore.ops.StopGradient](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.StopGradient.html#mindspore.ops.StopGradient)|Deleted|用于消除某个值对梯度的影响，例如截断来自于函数输出的梯度传播。|Ascend/GPU/CPU|框架算子|
|[mindspore.ops.GLU](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GLU.html#mindspore.ops.GLU)|New|门线性单元函数（Gated Linear Unit function）。|r2.0: Ascend/CPU|激活函数|
|[mindspore.ops.CTCGreedyDecoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.CTCGreedyDecoder.html#mindspore.ops.CTCGreedyDecoder)|Changed|对输入中给定的logits执行贪婪解码。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.DataFormatVecPermute](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.DataFormatVecPermute.html#mindspore.ops.DataFormatVecPermute)|Deleted|将输入按从 src_format 到 dst_format 的变化重新排列。|GPU/CPU|神经网络|
|[mindspore.ops.FractionalAvgPool](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.FractionalAvgPool.html#mindspore.ops.FractionalAvgPool)|Deleted|在输入上执行分数平均池化。|GPU/CPU|神经网络|
|[mindspore.ops.FractionalMaxPool](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.FractionalMaxPool.html#mindspore.ops.FractionalMaxPool)|Deleted|在输入上执行分数最大池化。|GPU/CPU|神经网络|
|[mindspore.ops.GridSampler2D](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GridSampler2D.html#mindspore.ops.GridSampler2D)|New|此操作使用基于流场网格的插值对2D input_x 进行采样，该插值通常由 mindspore.ops.affine_grid() 生成。|r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.GridSampler3D](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GridSampler3D.html#mindspore.ops.GridSampler3D)|New|给定一个输入和一个网格，使用网格中的输入值和像素位置计算输出。|r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.MaxPool3DWithArgmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxPool3DWithArgmax.html#mindspore.ops.MaxPool3DWithArgmax)|Changed|三维最大值池化，返回最大值结果及其索引值。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Changed|r2.0.0-alpha: 对输入Tensor执行最大池化运算，并返回最大值和索引。 => r2.0: ops.MaxPoolWithArgmax 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 ops.MaxPoolWithArgmaxV2 代替。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: |神经网络|
|[mindspore.ops.MaxPoolWithArgmaxV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxPoolWithArgmaxV2.html#mindspore.ops.MaxPoolWithArgmaxV2)|New|对输入Tensor执行最大池化运算，并返回最大值和索引。|r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.MaxUnpool2D](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxUnpool2D.html#mindspore.ops.MaxUnpool2D)|Changed|r2.0.0-alpha: 计算MaxPool2D的部分逆。 => r2.0: MaxPool2D的逆过程。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.MaxUnpool3D](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxUnpool3D.html#mindspore.ops.MaxUnpool3D)|New|mindspore.ops.MaxPool3D 的逆过程。|r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.MirrorPad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MirrorPad.html#mindspore.ops.MirrorPad)|Changed|通过指定的填充模式和大小对输入Tensor进行填充。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|神经网络|
|[mindspore.ops.UpsampleTrilinear3D](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.UpsampleTrilinear3D.html#mindspore.ops.UpsampleTrilinear3D)|Deleted|输入为五维度Tensor，跨其中三维执行三线性插值上调采样。|GPU/CPU|神经网络|
|[mindspore.ops.SparseTensorDenseMatmul](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SparseTensorDenseMatmul.html#mindspore.ops.SparseTensorDenseMatmul)|Changed|稀疏矩阵 A 乘以稠密矩阵 B 。|r2.0.0-alpha: CPU => r2.0: GPU/CPU|稀疏算子|
|[mindspore.ops.MatrixInverse](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MatrixInverse.html#mindspore.ops.MatrixInverse)|Changed|计算输入矩阵的逆矩阵。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|线性代数算子|
|[mindspore.ops.Orgqr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Orgqr.html#mindspore.ops.Orgqr)|New|计算 Householder 矩阵乘积的前 \(N\) 列。|r2.0: Ascend/GPU/CPU|线性代数算子|
|[mindspore.ops.Svd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Svd.html#mindspore.ops.Svd)|New|计算一个或多个矩阵的奇异值分解。|r2.0: GPU/CPU|线性代数算子|
|[mindspore.ops.Assert](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Assert.html#mindspore.ops.Assert)|Deleted|判断给定条件是否为True，若不为True则以list的形式打印 input_data 中的Tensor，否则继续往下运行代码。|GPU/CPU|调试算子|
|[mindspore.ops.Pdist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Pdist.html#mindspore.ops.Pdist)|New|计算输入中每对行向量之间的p-范数距离。|r2.0: GPU/CPU|距离函数|
|[mindspore.ops.AccumulateNV2](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AccumulateNV2.html#mindspore.ops.AccumulateNV2)|Changed|逐元素将所有输入的Tensor相加。|r2.0.0-alpha: Ascend => r2.0: Ascend/GPU|逐元素运算|
|[mindspore.ops.Angle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Angle.html#mindspore.ops.Angle)|New|逐元素计算复数Tensor的辐角。|r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.BesselI0](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BesselI0.html#mindspore.ops.BesselI0)|New|逐元素计算输入数据的BesselI0函数值。|r2.0: GPU/CPU|逐元素运算|
|[mindspore.ops.BesselI1](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BesselI1.html#mindspore.ops.BesselI1)|New|逐元素计算并返回输入Tensor的BesselI1函数值。|r2.0: GPU/CPU|逐元素运算|
|[mindspore.ops.Digamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Digamma.html#mindspore.ops.Digamma)|New|计算输入的lgamma函数的导数。|r2.0: GPU/CPU|逐元素运算|
|[mindspore.ops.Geqrf](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Geqrf.html#mindspore.ops.Geqrf)|New|将矩阵分解为正交矩阵 Q 和上三角矩阵 R 的乘积。|r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.Imag](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Imag.html#mindspore.ops.Imag)|Changed|返回包含输入Tensor的虚部。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.LogicalXor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LogicalXor.html#mindspore.ops.LogicalXor)|New|逐元素计算两个Tensor的逻辑异或运算。|r2.0: Ascend/CPU|逐元素运算|
|[mindspore.ops.Logit](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Logit.html#mindspore.ops.Logit)|New|逐元素计算Tensor的logit值。|r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.NextAfter](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.NextAfter.html#mindspore.ops.NextAfter)|Changed|逐元素返回 x1 指向 x2 的下一个可表示值符点值。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.Polar](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Polar.html#mindspore.ops.Polar)|New|将极坐标转化为笛卡尔坐标。|r2.0: GPU/CPU|逐元素运算|
|[mindspore.ops.Polygamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Polygamma.html#mindspore.ops.Polygamma)|New|计算关于 x 的多伽马函数的 \(a\) 阶导数。|r2.0: GPU/CPU|逐元素运算|
|[mindspore.ops.Real](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Real.html#mindspore.ops.Real)|Changed|返回输入Tensor的实数部分。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.Sinc](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sinc.html#mindspore.ops.Sinc)|New|逐元素计算输入Tensor的数学正弦函数。|r2.0: Ascend/GPU/CPU|逐元素运算|
|[mindspore.ops.UpsampleNearest3D](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.UpsampleNearest3D.html#mindspore.ops.UpsampleNearest3D)|Deleted|执行最近邻上采样操作。|GPU/CPU|采样算子|
|[mindspore.ops.Bernoulli](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Bernoulli.html#mindspore.ops.Bernoulli)|New|以 p 的概率随机将输出的元素设置为0或1，服从伯努利分布。|r2.0: GPU/CPU|随机生成算子|
|[mindspore.ops.Multinomial](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Multinomial.html#mindspore.ops.Multinomial)|Changed|返回从输入Tensor对应行进行多项式概率分布采样出的Tensor。|r2.0.0-alpha: GPU/CPU => r2.0: Ascend/GPU/CPU|随机生成算子|
|[mindspore.ops.MultinomialWithReplacement](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MultinomialWithReplacement.html#mindspore.ops.MultinomialWithReplacement)|New|返回一个Tensor，其中每行包含从重复采样的多项式分布中抽取的 numsamples 个索引。|r2.0: CPU|随机生成算子|
|[mindspore.ops.RandomGamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.RandomGamma.html#mindspore.ops.RandomGamma)|New|根据概率密度函数分布生成随机正值浮点数x。|r2.0: CPU|随机生成算子|
|[mindspore.ops.RandomPoisson](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.RandomPoisson.html#mindspore.ops.RandomPoisson)|New|根据离散概率密度函数分布生成随机非负数浮点数i。|r2.0: GPU/CPU|随机生成算子|
