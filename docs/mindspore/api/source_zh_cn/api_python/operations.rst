operations
----------

The Primitive operators in operations need to be instantiated before being used.

Neural Network Operators
^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Acosh
    mindspore.ops.Adam
    mindspore.ops.AdamNoUpdateParam
    mindspore.ops.AdamWeightDecay
    mindspore.ops.AdaptiveAvgPool2D
    mindspore.ops.ApplyAdadelta
    mindspore.ops.ApplyAdagrad
    mindspore.ops.ApplyAdagradDA
    mindspore.ops.ApplyAdagradV2
    mindspore.ops.ApplyAdaMax
    mindspore.ops.ApplyAddSign
    mindspore.ops.ApplyCenteredRMSProp
    mindspore.ops.ApplyGradientDescent
    mindspore.ops.ApplyMomentum
    mindspore.ops.ApplyPowerSign
    mindspore.ops.ApplyProximalAdagrad
    mindspore.ops.ApplyProximalGradientDescent
    mindspore.ops.ApplyRMSProp
    mindspore.ops.AvgPool
    mindspore.ops.AvgPool3D
    mindspore.ops.BasicLSTMCell
    mindspore.ops.BatchNorm
    mindspore.ops.BCEWithLogitsLoss
    mindspore.ops.BiasAdd
    mindspore.ops.BinaryCrossEntropy
    mindspore.ops.ComputeAccidentalHits
    mindspore.ops.Conv2D
    mindspore.ops.Conv2DBackpropInput
    mindspore.ops.Conv3D
    mindspore.ops.Conv3DTranspose
    mindspore.ops.CTCGreedyDecoder
    mindspore.ops.CTCLoss
    mindspore.ops.DataFormatDimMap
    mindspore.ops.DepthwiseConv2dNative
    mindspore.ops.Dropout
    mindspore.ops.Dropout2D
    mindspore.ops.Dropout3D
    mindspore.ops.DropoutDoMask
    mindspore.ops.DropoutGenMask
    mindspore.ops.DynamicGRUV2
    mindspore.ops.DynamicRNN
    mindspore.ops.Elu
    mindspore.ops.FastGeLU
    mindspore.ops.Flatten
    mindspore.ops.FloorMod
    mindspore.ops.FusedSparseAdam
    mindspore.ops.FusedSparseLazyAdam
    mindspore.ops.FusedSparseProximalAdagrad
    mindspore.ops.GeLU
    mindspore.ops.GetNext
    mindspore.ops.HShrink
    mindspore.ops.HSigmoid
    mindspore.ops.HSwish
    mindspore.ops.KLDivLoss
    mindspore.ops.L2Loss
    mindspore.ops.L2Normalize
    mindspore.ops.LARSUpdate
    mindspore.ops.LayerNorm
    mindspore.ops.LogSoftmax
    mindspore.ops.LRN
    mindspore.ops.LSTM
    mindspore.ops.MaxPool
    mindspore.ops.MaxPool3D
    mindspore.ops.MaxPoolWithArgmax
    mindspore.ops.MirrorPad
    mindspore.ops.Mish
    mindspore.ops.NLLLoss
    mindspore.ops.OneHot
    mindspore.ops.Pad
    mindspore.ops.PReLU
    mindspore.ops.ReLU
    mindspore.ops.ReLU6
    mindspore.ops.ReLUV2
    mindspore.ops.ResizeBilinear
    mindspore.ops.RNNTLoss
    mindspore.ops.ROIAlign
    mindspore.ops.SeLU
    mindspore.ops.SGD
    mindspore.ops.Sigmoid
    mindspore.ops.SigmoidCrossEntropyWithLogits
    mindspore.ops.SmoothL1Loss
    mindspore.ops.SoftMarginLoss
    mindspore.ops.Softmax
    mindspore.ops.SoftmaxCrossEntropyWithLogits
    mindspore.ops.Softplus
    mindspore.ops.SoftShrink
    mindspore.ops.Softsign
    mindspore.ops.SparseApplyAdagrad
    mindspore.ops.SparseApplyAdagradV2
    mindspore.ops.SparseApplyProximalAdagrad
    mindspore.ops.SparseSoftmaxCrossEntropyWithLogits
    mindspore.ops.Stack
    mindspore.ops.Tanh
    mindspore.ops.TopK
    mindspore.ops.Unstack

Math Operators
^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Abs
    mindspore.ops.AccumulateNV2
    mindspore.ops.ACos
    mindspore.ops.Add
    mindspore.ops.AddN
    mindspore.ops.ApproximateEqual
    mindspore.ops.Asin
    mindspore.ops.Asinh
    mindspore.ops.AssignAdd
    mindspore.ops.AssignSub
    mindspore.ops.Atan
    mindspore.ops.Atan2
    mindspore.ops.Atanh
    mindspore.ops.BatchMatMul
    mindspore.ops.BesselI0e
    mindspore.ops.BesselI1e
    mindspore.ops.BitwiseAnd
    mindspore.ops.BitwiseOr
    mindspore.ops.BitwiseXor
    mindspore.ops.Ceil
    mindspore.ops.Cos
    mindspore.ops.Conj
    mindspore.ops.Cosh
    mindspore.ops.CumProd
    mindspore.ops.CumSum
    mindspore.ops.Div
    mindspore.ops.DivNoNan
    mindspore.ops.Eps
    mindspore.ops.Equal
    mindspore.ops.EqualCount
    mindspore.ops.Erf
    mindspore.ops.Erfc
    mindspore.ops.Erfinv
    mindspore.ops.Exp
    mindspore.ops.Expm1
    mindspore.ops.FloatStatus
    mindspore.ops.Floor
    mindspore.ops.FloorDiv
    mindspore.ops.Greater
    mindspore.ops.GreaterEqual
    mindspore.ops.HistogramFixedWidth
    mindspore.ops.Imag
    mindspore.ops.IndexAdd
    mindspore.ops.InplaceAdd
    mindspore.ops.InplaceSub
    mindspore.ops.Inv
    mindspore.ops.Invert
    mindspore.ops.IsInf
    mindspore.ops.IsNan
    mindspore.ops.Lerp
    mindspore.ops.Less
    mindspore.ops.LessEqual
    mindspore.ops.LinSpace
    mindspore.ops.Log
    mindspore.ops.Log1p
    mindspore.ops.LogicalAnd
    mindspore.ops.LogicalNot
    mindspore.ops.LogicalOr
    mindspore.ops.MatMul
    mindspore.ops.MatrixInverse
    mindspore.ops.Maximum
    mindspore.ops.Minimum
    mindspore.ops.Mod
    mindspore.ops.Mul
    mindspore.ops.MulNoNan
    mindspore.ops.Neg
    mindspore.ops.NMSWithMask
    mindspore.ops.NotEqual
    mindspore.ops.NPUAllocFloatStatus
    mindspore.ops.NPUClearFloatStatus
    mindspore.ops.NPUGetFloatStatus
    mindspore.ops.Pow
    mindspore.ops.Real
    mindspore.ops.RealDiv
    mindspore.ops.Reciprocal
    mindspore.ops.ReduceAll
    mindspore.ops.ReduceAny
    mindspore.ops.ReduceMax
    mindspore.ops.ReduceMean
    mindspore.ops.ReduceMin
    mindspore.ops.ReduceProd
    mindspore.ops.ReduceSum
    mindspore.ops.Round
    mindspore.ops.Rsqrt
    mindspore.ops.Sign
    mindspore.ops.Sin
    mindspore.ops.Sinh
    mindspore.ops.Sqrt
    mindspore.ops.Square
    mindspore.ops.SquaredDifference
    mindspore.ops.SquareSumAll
    mindspore.ops.Sub
    mindspore.ops.Tan
    mindspore.ops.TruncateDiv
    mindspore.ops.TruncateMod
    mindspore.ops.Xdivy
    mindspore.ops.Xlogy

Array Operators
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.ApplyFtrl
    mindspore.ops.Argmax
    mindspore.ops.ArgMaxWithValue
    mindspore.ops.Argmin
    mindspore.ops.ArgMinWithValue
    mindspore.ops.BatchToSpace
    mindspore.ops.BatchToSpaceND
    mindspore.ops.BroadcastTo
    mindspore.ops.Cast
    mindspore.ops.Concat
    mindspore.ops.DepthToSpace
    mindspore.ops.DType
    mindspore.ops.DynamicShape
    mindspore.ops.EditDistance
    mindspore.ops.EmbeddingLookup
    mindspore.ops.ExpandDims
    mindspore.ops.Eye
    mindspore.ops.Fill
    mindspore.ops.FusedSparseFtrl
    mindspore.ops.Gather
    mindspore.ops.GatherD
    mindspore.ops.GatherNd
    mindspore.ops.Identity
    mindspore.ops.InplaceUpdate
    mindspore.ops.InvertPermutation
    mindspore.ops.IsFinite
    mindspore.ops.IsInstance
    mindspore.ops.IsSubClass
    mindspore.ops.MaskedFill
    mindspore.ops.MaskedSelect
    mindspore.ops.Meshgrid
    mindspore.ops.Ones
    mindspore.ops.OnesLike
    mindspore.ops.Padding
    mindspore.ops.ParallelConcat
    mindspore.ops.Randperm
    mindspore.ops.Rank
    mindspore.ops.Reshape
    mindspore.ops.ResizeNearestNeighbor
    mindspore.ops.ReverseSequence
    mindspore.ops.ReverseV2
    mindspore.ops.Rint
    mindspore.ops.SameTypeShape
    mindspore.ops.ScalarCast
    mindspore.ops.ScalarToArray
    mindspore.ops.ScalarToTensor
    mindspore.ops.ScatterAdd
    mindspore.ops.ScatterDiv
    mindspore.ops.ScatterMax
    mindspore.ops.ScatterMin
    mindspore.ops.ScatterMul
    mindspore.ops.ScatterNd
    mindspore.ops.ScatterNdAdd
    mindspore.ops.ScatterNdSub
    mindspore.ops.ScatterNdUpdate
    mindspore.ops.ScatterNonAliasingAdd
    mindspore.ops.ScatterSub
    mindspore.ops.ScatterUpdate
    mindspore.ops.Select
    mindspore.ops.Shape
    mindspore.ops.Size
    mindspore.ops.Slice
    mindspore.ops.Sort
    mindspore.ops.SpaceToBatch
    mindspore.ops.SpaceToBatchND
    mindspore.ops.SpaceToDepth
    mindspore.ops.SparseApplyFtrl
    mindspore.ops.SparseApplyFtrlV2
    mindspore.ops.SparseGatherV2
    mindspore.ops.Split
    mindspore.ops.SplitV
    mindspore.ops.Squeeze
    mindspore.ops.StridedSlice
    mindspore.ops.TensorScatterAdd
    mindspore.ops.TensorScatterMax
    mindspore.ops.TensorScatterMin
    mindspore.ops.TensorScatterSub
    mindspore.ops.TensorScatterUpdate
    mindspore.ops.Tile
    mindspore.ops.Transpose
    mindspore.ops.TupleToArray
    mindspore.ops.Unique
    mindspore.ops.UniqueWithPad
    mindspore.ops.UnsortedSegmentMax
    mindspore.ops.UnsortedSegmentMin
    mindspore.ops.UnsortedSegmentProd
    mindspore.ops.UnsortedSegmentSum
    mindspore.ops.Zeros
    mindspore.ops.ZerosLike

Communication Operators
^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AllGather
    mindspore.ops.AllReduce
    mindspore.ops.Broadcast
    mindspore.ops.ReduceOp
    mindspore.ops.ReduceScatter

Debug Operators
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.HistogramSummary
    mindspore.ops.HookBackward
    mindspore.ops.ImageSummary
    mindspore.ops.InsertGradientOf
    mindspore.ops.Print
    mindspore.ops.ScalarSummary
    mindspore.ops.TensorSummary

Random Operators
^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Gamma
    mindspore.ops.LogUniformCandidateSampler
    mindspore.ops.Multinomial
    mindspore.ops.Poisson
    mindspore.ops.RandomCategorical
    mindspore.ops.RandomChoiceWithMask
    mindspore.ops.StandardLaplace
    mindspore.ops.StandardNormal
    mindspore.ops.UniformCandidateSampler
    mindspore.ops.UniformInt
    mindspore.ops.UniformReal

Image Operators
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.CropAndResize

Sparse Operators
^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.SparseToDense
    mindspore.ops.SparseTensorDenseMatmul

Other Operators
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Assign
    mindspore.ops.BoundingBoxDecode
    mindspore.ops.BoundingBoxEncode
    mindspore.ops.CheckValid
    mindspore.ops.Depend
    mindspore.ops.InTopK
    mindspore.ops.IOU
    mindspore.ops.NoRepeatNGram
    mindspore.ops.PopulationCount
