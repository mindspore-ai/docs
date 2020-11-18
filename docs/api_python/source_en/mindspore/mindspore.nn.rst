mindspore.nn
============

.. automodule:: mindspore.nn

Cell
----

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Cell
    mindspore.nn.GraphKernel

Containers
----------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CellList
    mindspore.nn.SequentialCell

Convolution Layers
------------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Conv1d
    mindspore.nn.Conv1dTranspose
    mindspore.nn.Conv2d
    mindspore.nn.Conv2dTranspose

Recurrent Layers
----------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.LSTMCell
    mindspore.nn.LSTM

Sparse Layers
-------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Embedding
    mindspore.nn.EmbeddingLookup

Non-linear Activations
----------------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.GELU
    mindspore.nn.HSigmoid
    mindspore.nn.HSwish
    mindspore.nn.LeakyReLU
    mindspore.nn.LogSigmoid
    mindspore.nn.LogSoftmax
    mindspore.nn.PReLU
    mindspore.nn.ReLU
    mindspore.nn.ReLU6
    mindspore.nn.Sigmoid
    mindspore.nn.Softmax
    mindspore.nn.Tanh

Utilities
---------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ClipByNorm
    mindspore.nn.Dense
    mindspore.nn.Dropout
    mindspore.nn.Flatten
    mindspore.nn.Norm
    mindspore.nn.OneHot
    mindspore.nn.Pad
    mindspore.nn.Range
    mindspore.nn.Unfold

Images Functions
----------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CentralCrop
    mindspore.nn.ImageGradients
    mindspore.nn.MSSSIM
    mindspore.nn.PSNR
    mindspore.nn.SSIM

Normalization Layers
--------------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BatchNorm1d
    mindspore.nn.BatchNorm2d
    mindspore.nn.GlobalBatchNorm
    mindspore.nn.GroupNorm
    mindspore.nn.LayerNorm
    mindspore.nn.LinSpace
    mindspore.nn.MatrixDiag
    mindspore.nn.MatrixDiagPart
    mindspore.nn.MatrixSetDiag

Pooling layers
--------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.AvgPool1d
    mindspore.nn.AvgPool2d
    mindspore.nn.MaxPool1d
    mindspore.nn.MaxPool2d

Quantized Functions
-------------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ActQuant
    mindspore.nn.Conv2dBnAct
    mindspore.nn.Conv2dBnFoldQuant
    mindspore.nn.Conv2dBnWithoutFoldQuant
    mindspore.nn.Conv2dQuant
    mindspore.nn.DenseBnAct
    mindspore.nn.DenseQuant
    mindspore.nn.MulQuant
    mindspore.nn.TensorAddQuant

Loss Functions
--------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BCELoss
    mindspore.nn.CosineEmbeddingLoss
    mindspore.nn.L1Loss
    mindspore.nn.MSELoss
    mindspore.nn.SmoothL1Loss
    mindspore.nn.SoftmaxCrossEntropyWithLogits

Optimizer Functions
-------------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Adam
    mindspore.nn.AdamWeightDecay
    mindspore.nn.FTRL
    mindspore.nn.Lamb
    mindspore.nn.LARS
    mindspore.nn.LazyAdam
    mindspore.nn.Momentum
    mindspore.nn.Optimizer
    mindspore.nn.ProximalAdagrad
    mindspore.nn.RMSProp
    mindspore.nn.SGD

Wrapper Functions
-----------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.DistributedGradReducer
    mindspore.nn.DynamicLossScaleUpdateCell
    mindspore.nn.FixedLossScaleUpdateCell
    mindspore.nn.GetNextSingleOp
    mindspore.nn.ParameterUpdate
    mindspore.nn.TrainOneStepCell
    mindspore.nn.TrainOneStepWithLossScaleCell
    mindspore.nn.WithEvalCell
    mindspore.nn.WithGradCell
    mindspore.nn.WithLossCell

Math Functions
--------------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.LGamma
    mindspore.nn.MatMul
    mindspore.nn.Moments
    mindspore.nn.ReduceLogSumExp
