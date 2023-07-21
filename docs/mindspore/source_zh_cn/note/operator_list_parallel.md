# MindSpore分布式API支持

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/operator_list_parallel.md)

| 操作名                                                       | 约束                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [mindspore.ops.Abs](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Abs.html) | 无                                                           |
| [mindspore.ops.ACos](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ACos.html) | 无                                                           |
| [mindspore.ops.Acosh](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Acosh.html) | 无                                                           |
| [mindspore.ops.Add](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Add.html) | 无                                                           |
| [mindspore.ops.AddN](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AddN.html) | 无                                                           |
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ApproximateEqual.html) | 无                                                           |
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ArgMaxWithValue.html) | 输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致 |
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ArgMinWithValue.html) | 输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致 |
| [mindspore.ops.Asin](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Asin.html) | 无                                                           |
| [mindspore.ops.Asinh](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Asinh.html) | 无                                                           |
| [mindspore.ops.Assign](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Assign.html) | 无                                                           |
| [mindspore.ops.AssignAdd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AssignAdd.html) | 无                                                           |
| [mindspore.ops.AssignSub](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AssignSub.html) | 无                                                           |
| [mindspore.ops.Atan](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Atan.html) | 无                                                           |
| [mindspore.ops.Atan2](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Atan2.html) | 无                                                           |
| [mindspore.ops.Atanh](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Atanh.html) | 无                                                           |
| [mindspore.ops.AvgPool](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AvgPool.html) | 1. 数据格式只支持‘NCHW’；<br />2. 输出的H/W维的shape必须能被输入的H/W维的切分策略整除；<br />3. 如果切分H/W：<br />     1）当kernel_size <= stride时，输入切片大小需能被stride整除；<br />     2）不支持kernel_size > stride；<br />4. 在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.AvgPool3D](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.AvgPool3D.html) | 1. 数据格式只支持‘NCDHW’；<br />2. 如果涉及相邻节点数据交换，只支持Ascend；<br />3. W维不支持切分；<br />4. 输出的D/H维的shape必须能被输入的D/H维的切分策略整除；<br />5. valid模式下：如果切分D/H：<br />     1）当kernel_size <= stride时，输入切片大小需能被stride整除；<br />     2）不支持kernel_size > stride；<br />6. same/pad模式下：如果切分D/H：<br />     1）kernel_size >= stride时，（包含pad的输入总长度 - kernel_size）需能被stride整除；kernel_size < stride时，pad需等于0且分片能被stride整除；<br />     2）（输出总长度*stride - 输入总长度）需能被切分策略整除；<br />     3）相邻卡间发送接收的数据长度需大于等于0且小于等于切片大小；<br />7.  在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BatchMatMul.html) | 不支持`transpose_a=True`                                     |
| [mindspore.ops.BatchNorm](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BatchNorm.html) | 不支持GPU                                                    |
| [mindspore.ops.BesselI0e](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BesselI0e.html) | 无                                                           |
| [mindspore.ops.BesselI1e](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BesselI1e.html) | 无                                                           |
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BiasAdd.html) | 无                                                           |
| [mindspore.ops.BitwiseAnd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BitwiseAnd.html) | 无                                                           |
| [mindspore.ops.BitwiseOr](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BitwiseOr.html) | 无                                                           |
| [mindspore.ops.BitwiseXor](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BitwiseXor.html) | 无                                                           |
| [mindspore.ops.BoundingBoxEncode](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BoundingBoxEncode.html) | 1. 支持对输入（anchor_box）和输入（groundtruth_box）的第一维进行切分； <br /> 2. 输入（anchor_box）和输入（groundtruth_box）的切分策略必须一致 |
| [mindspore.ops.BroadcastTo](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.BroadcastTo.html) | 无                                                           |
| [mindspore.ops.Cast](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cast.html) | Auto Parallel和Semi Auto Parallel模式下，配置策略不生效      |
| [mindspore.ops.Cdist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cdist.html) | 1. 当两输入都含有Batch维时，这一维的切分策略必须相等；<br /> 2.`M`维度不支持切分 |
| [mindspore.ops.Ceil](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Ceil.html) | 无                                                           |
| [mindspore.ops.Concat](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Concat.html) | 输入（input_x）在轴（axis）所对应的维度不能切分，切分后，在数学逻辑上和单机不等价 |
| [mindspore.ops.Conv2D](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Conv2D.html) | 1. 数据格式只支持‘NCHW’；<br />2. 如果涉及相邻节点数据交换，只支持Ascend；<br />3. 当group的值不为1时，不支持切分C-in/C-out；<br />4. weight的后两维不支持切分；<br />5. 输出的H/W维的shape必须能被输入的H/W维的切分策略整除；<br />6. valid模式下：如果切分H/W：<br />     1）当kernel_size <= stride时其中（kernel_size=dilation *(kernel_size - 1) + 1，下同），输入切片大小需能被stride整除；<br />     2）不支持kernel_size > stride；<br />7. same/pad模式下：如果切分H/W：<br />     1）kernel_size >= stride时，（包含pad的输入总长度 - kernel_size）需能被stride整除；kernel_size < stride时，pad需等于0且分片能被stride整除；<br />     2）（ 输出总长度*stride - 输入总长度）需能被切分策略整除；<br />     3）相邻卡间发送接收的数据长度需大于等于0且小于等于切片大小； |
| [mindspore.ops.Conv3D](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Conv3D.html) | 1. 数据格式只支持‘NCDHW’；<br />2. 如果涉及相邻节点数据交换，只支持Ascend；<br />3. 当group的值不为1时，不支持切分C-in/C-out；<br />4. W维不支持切分，weight的后三维不支持切分；<br />5. 输出的D/H维的shape必须能被输入的D/H维的切分策略整除；<br />6. valid模式下：如果切分D/H：<br />     1）当kernel_size <= stride时（kernel_size=dilation *(kernel_size - 1) + 1，下同），输入切片大小需能被stride整除；<br />     2）不支持kernel_size > stride；<br />7. same/pad模式下：如果切分D/H：<br />     1）kernel_size >= stride时，（包含pad的输入总长度 - kernel_size）需能被stride整除；kernel_size < stride时，pad需等于0且分片能被stride整除；<br />     2）（ 输出总长度*stride - 输入总长度）需能被切分策略整除；<br />     3）相邻卡间发送接收的数据长度需大于等于0且小于等于切片大小；<br />8.  在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.Cos](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cos.html) | 无                                                           |
| [mindspore.ops.Cosh](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Cosh.html) | 无                                                           |
| [mindspore.ops.CropAndResize](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.CropAndResize.html) | 1. 不支持对输入（x）的H/W维和输入（boxes）的第二维进行切分；<br /> 2. 输入（boxes）和输入（box_index）第一维的切分策略必须一致 |
| [mindspore.ops.CumProd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.CumProd.html) | 不支持`axis维`切分                                           |
| [mindspore.ops.CumSum](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.CumSum.html) | 不支持`axis维`切分                                           |
| [mindspore.ops.Div](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Div.html) | 无                                                           |
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.DivNoNan.html) | 无                                                           |
| [mindspore.ops.Dropout](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Dropout.html) | 无                                                           |
| [mindspore.ops.Elu](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Elu.html) | 无                                                           |
| [mindspore.ops.EmbeddingLookup](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.EmbeddingLookup.html) | 同Gather                                                     |
| [mindspore.ops.Equal](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Equal.html) | 无                                                           |
| [mindspore.ops.Erf](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Erf.html) | 无                                                           |
| [mindspore.ops.Erfc](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Erfc.html) | 无                                                           |
| [mindspore.ops.Erfinv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Erfinv.html) | 无                                                           |
| [mindspore.ops.Exp](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Exp.html) | 无                                                           |
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ExpandDims.html) | 无                                                           |
| [mindspore.ops.Expm1](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Expm1.html) | 无                                                           |
| [mindspore.ops.Floor](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Floor.html) | 无                                                           |
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.FloorDiv.html) | 无                                                           |
| [mindspore.ops.FloorMod](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.FloorMod.html) | 无                                                           |
| [mindspore.ops.Gamma](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Gamma.html) | 1. `shape`为tuple，但可以对`shape`进行切分，如shape=(8, 16)，对应的策略可以为(2, 4); <br /> 2. `alpha`和`beta`对应的策略必须为全1; <br /> 3. 当`shape`切分策略不为全1时，结果和单卡不一致 |
| [mindspore.ops.Gather](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Gather.html) | 1. 均匀切分时：<br />1）仅支持一维和二维的input_params，并且input_params的切片的最后一维要32字节对齐（出于性能考虑）；<br />2）不支持标量input_indices；<br />3）参数在轴（axis）所在维度切分时，不支持重复计算；<br />4）不支持input_indices和input_params同时进行切分；<br />5）axis=0且参数在轴（axis）所在维度切分时，支持配置输出切分策略，合法的输出切分策略为(indices_strategy, param_strategy[1:]) 或 ((indices_strategy[0]*param_strategy[0], indices_strategy[1:]), param_strategy[1:])<br />2. 非均匀切分时：<br />1）axis仅支持为0；<br />2）非均匀切分仅表示对input_params的第零维非均匀切分，并且input_params的切片的最后一维要32字节对齐；<br />3）对input_params第零维的切分份数要等于对input_indices最后一维的切分份数；<br />4）input_params的每个维度都能切分，但input_indices只能切分最后一维，且不支持重复计算；<br />5）input_indices需满足：后一个切片的Tensor值需大于前一个分片的Tensor值。 |
| [mindspore.ops.GatherD](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GatherD.html) | dim所对应的维度不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.GatherNd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GatherNd.html) | 第一个输入不能切分，第二个输入的最后一维不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.GeLU](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GeLU.html) | 无                                                           |
| [mindspore.ops.Greater](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Greater.html) | 无                                                           |
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.GreaterEqual.html) | 无                                                           |
| [mindspore.ops.HShrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.HShrink.html) | 无                                                           |
| [mindspore.ops.HSigmoid](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.HSigmoid.html) | 无                                                           |
| [mindspore.ops.InplaceAdd](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceAdd.html) | 不支持对`x`和`input_v`的第一维切分                           |
| [mindspore.ops.InplaceSub](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceSub.html) | 同InplaceAdd                                                 |
| [mindspore.ops.InplaceUpdate](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.InplaceUpdate.html) | 同InplaceAdd                                                 |
| [mindspore.ops.Inv](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Inv.html) | 无                                                           |
| [mindspore.ops.IOU](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.IOU.html) | 支持对`anchor_boxes`和`gt_boxes`的第一维切分                 |
| [mindspore.ops.IsFinite](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.IsFinite.html) | 无                                                           |
| [mindspore.ops.KLDivLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.KLDivLoss.html) | 无                                                           |
| [mindspore.ops.L2Loss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.L2Loss.html) | 无                                                           |
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.L2Normalize.html) | 输入（input_x）在轴（axis）对应的维度不能切，切分后，在数学逻辑上和单机不等价 |
| [mindspore.ops.Lerp](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Lerp.html) | 无                                                           |
| [mindspore.ops.Less](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Less.html) | 无                                                           |
| [mindspore.ops.LessEqual](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LessEqual.html) | 无                                                           |
| [mindspore.ops.LinSpace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LinSpace.html) | 不需要为`start`和`end`配置策略，只需要传入一个长度为1的策略，其数值能整除`num` |
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LogicalAnd.html) | 无                                                           |
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LogicalNot.html) | 无                                                           |
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LogicalOr.html) | 无                                                           |
| [mindspore.ops.Log](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Log.html) | 无                                                           |
| [mindspore.ops.Log1p](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Log1p.html) | 无                                                           |
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.LogSoftmax.html) | 输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价 |
| [mindspore.ops.MaskedFill](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaskedFill.html) | 无                                                           |
| [mindspore.ops.MatMul](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MatMul.html) | 1. 不支持`transpose_a=True`；<br />2. 当`transpose_b=True`时，输入的切分策略需是 ((A, B), (C, B)) 的形式<br />3. 当`transpose_b=False`时，输入的切分策略需是 ((A, B), (B, C)) 的形式；<br />4. 支持设置输出切分策略，合法的输出切分策略为 ((A, C),) 或 ((A * B, C),) 。 |
| [mindspore.ops.Maximum](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Maximum.html) | 无                                                           |
| [mindspore.ops.MaxPool](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxPool.html) | 1. 数据格式只支持‘NCHW’；<br />2. 输出的H/W维的shape必须能被输入的H/W维的切分策略整除；<br />3. 如果切分H/W：<br />     1）当kernel_size <= stride时，输入切片大小需能被stride整除；<br />     2）不支持kernel_size > stride；<br />4. 在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.MaxPool3D](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MaxPool3D.html) | 同AvgPool3D                                                  |
| [mindspore.ops.Minimum](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Minimum.html) | 无                                                           |
| [mindspore.ops.Mish](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Mish.html) | 无                                                           |
| [mindspore.ops.Mod](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Mod.html) | 无                                                           |
| [mindspore.ops.Mul](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Mul.html) | 无                                                           |
| [mindspore.ops.MulNoNan](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.MulNoNan.html) | 无                                                           |
| [mindspore.ops.Neg](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Neg.html) | 无                                                           |
| [mindspore.ops.NotEqual](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.NotEqual.html) | 无                                                           |
| [mindspore.ops.OneHot](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.OneHot.html) | 仅支持输入（indices）是一维的Tensor，切分策略要配置输出的切分策略，以及第1和第2个输入的切分策略 |
| [mindspore.ops.OnesLike](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.OnesLike.html) | 无                                                           |
| [mindspore.ops.Pow](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Pow.html) | 无                                                           |
| [mindspore.ops.PReLU](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.PReLU.html) | weight的shape在非[1]的情况下，输入（input_x）的Channel维要和weight的切分方式一致 |
| [mindspore.ops.RandomChoiceWithMask](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.RandomChoiceWithMask.html) | 不支持切分，仅支持全1策略                                    |
| [mindspore.ops.RealDiv](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.RealDiv.html) | 无                                                           |
| [mindspore.ops.Reciprocal](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Reciprocal.html) | 无                                                           |
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ReduceMax.html) | 输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致 |
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ReduceMin.html) | 输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致 |
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ReduceSum.html) | 无                                                           |
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ReduceMean.html) | 无                                                           |
| [mindspore.ops.ReLU](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ReLU.html) | 无                                                           |
| [mindspore.ops.ReLU6](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ReLU6.html) | 无                                                           |
| [mindspore.ops.Reshape](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Reshape.html) | 不支持配置切分策略，并且，在自动并行模式下，当reshape API后接有多个API，不允许对这些API配置不同的切分策略 |
| [mindspore.ops.ResizeBilinear](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ResizeBilinear.html) | 在GPU平台下，不支持H/W维切分；在Ascend平台下，不支持H维切分，且W维的输出shape要能被切分数整除。 |
| [mindspore.ops.ResizeNearestNeighbor](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ResizeNearestNeighbor.html) | 在`align_corners=True`时只支持切分第一维和第二维。           |
| [mindspore.ops.Rint](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Rint.html) | 无                                                           |
| [mindspore.ops.ROIAlign](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ROIAlign.html) | 不支持对输入（features）的H/W维和输入（rois）的第二维进行切分 |
| [mindspore.ops.Round](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Round.html) | 无                                                           |
| [mindspore.ops.Rsqrt](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Rsqrt.html) | 无                                                           |
| [mindspore.ops.ScatterAdd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterAdd.html) | 第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterDiv](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterDiv.html) | 第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterMax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterMax.html) | 第一个输入的第一维不能切分，第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterMin](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterMin.html) | 第一个输入的第一维不能切分，第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterMul](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterMul.html) | 第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterNdAdd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterNdAdd.html) | 第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterNdSub](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterNdSub.html) | 第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterNdUpdate](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterNdUpdate.html) | 第一个输入前m维度不能切（m为第二个输入indices的最后一维的值indices[-1]）第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterSub](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterSub.html) | 第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ScatterUpdate.html) | 第一个输入的第一维不能切分，第二个输入不能切分，第三个输入的前n维（n为第二个输入的维度）不能切分；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterAdd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterAdd.html) | 第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterDiv](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterDiv.html) | 第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterMax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterMax.html) | 第一个输入前m维度不能切（m为第二个输入indices的最后一维的值indices[-1]）第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterMax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterMin.html) | 第一个输入前m维度不能切（m为第二个输入indices的最后一维的值indices[-1]）第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterMul](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterMul.html) | 第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterAdd](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterSub.html) | 第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.TensorScatterUpdate](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TensorScatterUpdate.html) | 第一个输入前m维度不能切（m为第二个输入indices的最后一维的值indices[-1]）第二个输入不能切分，第三个输入的前n-1维（n为第二个输入的维度）不能切分，第三个输入的剩下k个维度（除去前n-1维度）的切分与第一个输入的最后k个切分一致；在auto_parallel模式下，不支持双递归算法。 |
| [mindspore.ops.Select](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Select.html) | 在auto_parallel模式下，不支持双递归算法。                    |
| [mindspore.ops.SeLU](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SeLU.html) | 无                                                           |
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sigmoid.html) | 无                                                           |
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SigmoidCrossEntropyWithLogits.html) | 无                                                           |
| [mindspore.ops.Sign](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sign.html) | 无                                                           |
| [mindspore.ops.Sin](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sin.html) | 无                                                           |
| [mindspore.ops.Sinh](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sinh.html) | 无                                                           |
| [mindspore.ops.Softmax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Softmax.html) | 输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价 |
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SoftmaxCrossEntropyWithLogits.html) | 输入（logits、labels）的最后一维不能切分；有两个输出，正向的loss只支持取[0] |
| [mindspore.ops.Softplus](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Softplus.html) | 无                                                           |
| [mindspore.ops.Softsign](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Softsign.html) | 无                                                           |
| [mindspore.ops.SoftShrink](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SoftShrink.html) | 无                                                           |
| [mindspore.ops.SparseGatherV2](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SparseGatherV2.html) | 同Gather                                                     |
| [mindspore.ops.Split](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Split.html) | 轴（axis）所对应的维度不能切分，切分后，在数学逻辑上和单机不等价 |
| [mindspore.ops.Sqrt](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sqrt.html) | 无                                                           |
| [mindspore.ops.Square](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Square.html) | 无                                                           |
| [mindspore.ops.SquaredDifference](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SquaredDifference.html) | 无                                                           |
| [mindspore.ops.Squeeze](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Squeeze.html) | 无                                                           |
| [mindspore.ops.Stack](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Stack.html) | 无                                                           |
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.StridedSlice.html) | 仅支持值为全0的mask；需要切分的维度必须全部提取；输入在strides不为1对应的维度不支持切分 |
| [mindspore.ops.Slice](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Slice.html) | 需要切分的维度必须全部提取                                   |
| [mindspore.ops.Sub](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Sub.html) | 无                                                           |
| [mindspore.ops.Tan](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Tan.html) | 无                                                           |
| [mindspore.ops.Tanh](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Tanh.html) | 无                                                           |
| [mindspore.ops.Tile](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Tile.html) | 仅支持对multiples配置切分策略                                |
| [mindspore.ops.TopK](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TopK.html) | 最后一维不支持切分，切分后，在数学逻辑上和单机不等价         |
| [mindspore.ops.Transpose](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Transpose.html) | 无                                                           |
| [mindspore.ops.TruncateDiv](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TruncateDiv.html) | 无                                                           |
| [mindspore.ops.TruncateMod](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.TruncateMod.html) | 无                                                           |
| [mindspore.ops.Unique](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Unique.html) | 只支持重复计算的策略(1,)                                     |
| [mindspore.ops.UnsortedSegmentSum](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.UnsortedSegmentSum.html) | 输入input_x和segment_ids的切分配置必须在segment_ids的维度上保持一致 |
| [mindspore.ops.UnsortedSegmentMin](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.UnsortedSegmentMin.html) | 输入input_x和segment_ids的切分配置必须在segment_ids的维度上保持一致。注意：在segment id为空时，输出向量的对应位置会填充为输入类型的最大值。需要用户进行掩码处理，将最大值转换成0。否则容易造成数值溢出，导致通信API上溢错误，从而引发Run Task Error |
| [mindspore.ops.UnsortedSegmentMax](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.UnsortedSegmentMax.html) | 输入input_x和segment_ids的切分配置必须在segment_ids的维度上保持一致。注意：在segment id为空时，输出向量的对应位置会填充为输入类型的最小值。需要用户进行掩码处理，将最小值转换成0。否则容易造成数值溢出，导致通信API上溢错误，从而引发Run Task Error |
| [mindspore.ops.Xdivy](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Xdivy.html) | 无                                                           |
| [mindspore.ops.Xlogy](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.Xlogy.html) | 无                                                           |
| [mindspore.ops.ZerosLike](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ZerosLike.html) | 无                                                           |

> 重复计算是指，机器没有用满，比如：集群有8张卡跑分布式训练，切分策略只对输入切成了4份。这种情况下会发生重复计算。
