mindspore.nn
============

.. automodule:: mindspore.nn

Cell
----

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Cell
    mindspore.nn.GraphKernel

Containers
----------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CellList
    mindspore.nn.SequentialCell

Convolution Layers
------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Conv1d
    mindspore.nn.Conv1dTranspose
    mindspore.nn.Conv2d
    mindspore.nn.Conv2dTranspose

Recurrent Layers
----------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.LSTMCell
    mindspore.nn.LSTM

Sparse Layers
-------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Embedding
    mindspore.nn.EmbeddingLookup
    mindspore.nn.MultiFieldEmbeddingLookup

Non-linear Activations
----------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ELU
    mindspore.nn.FastGelu
    mindspore.nn.GELU
    mindspore.nn.get_activation
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

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ClipByNorm
    mindspore.nn.Dense
    mindspore.nn.Dropout
    mindspore.nn.Flatten
    mindspore.nn.L1Regularizer
    mindspore.nn.Norm
    mindspore.nn.OneHot
    mindspore.nn.Pad
    mindspore.nn.Range
    mindspore.nn.ResizeBilinear
    mindspore.nn.Tril
    mindspore.nn.Triu
    mindspore.nn.Unfold

Images Functions
----------------

.. msplatformautosummary::
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

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BatchNorm1d
    mindspore.nn.BatchNorm2d
    mindspore.nn.GlobalBatchNorm
    mindspore.nn.GroupNorm
    mindspore.nn.LayerNorm
    mindspore.nn.MatrixDiag
    mindspore.nn.MatrixDiagPart
    mindspore.nn.MatrixSetDiag

Pooling layers
--------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.AvgPool1d
    mindspore.nn.AvgPool2d
    mindspore.nn.MaxPool1d
    mindspore.nn.MaxPool2d

Quantized Functions
-------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ActQuant
    mindspore.nn.Conv2dBnAct
    mindspore.nn.Conv2dBnFoldQuant
    mindspore.nn.Conv2dBnFoldQuantOneConv
    mindspore.nn.Conv2dBnWithoutFoldQuant
    mindspore.nn.Conv2dQuant
    mindspore.nn.DenseBnAct
    mindspore.nn.DenseQuant
    mindspore.nn.FakeQuantWithMinMaxObserver
    mindspore.nn.MulQuant
    mindspore.nn.TensorAddQuant

Loss Functions
--------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BCELoss
    mindspore.nn.CosineEmbeddingLoss
    mindspore.nn.L1Loss
    mindspore.nn.MSELoss
    mindspore.nn.SampledSoftmaxLoss
    mindspore.nn.SmoothL1Loss
    mindspore.nn.SoftmaxCrossEntropyWithLogits

Optimizer Functions
-------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Adagrad
    mindspore.nn.Adam
    mindspore.nn.AdamOffload
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

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.DistributedGradReducer
    mindspore.nn.DynamicLossScaleUpdateCell
    mindspore.nn.FixedLossScaleUpdateCell
    mindspore.nn.GetNextSingleOp
    mindspore.nn.ParameterUpdate
    mindspore.nn.TimeDistributed
    mindspore.nn.TrainOneStepCell
    mindspore.nn.TrainOneStepWithLossScaleCell
    mindspore.nn.WithEvalCell
    mindspore.nn.WithGradCell
    mindspore.nn.WithLossCell

Math Functions
--------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.DiGamma
    mindspore.nn.IGamma
    mindspore.nn.LBeta
    mindspore.nn.LGamma
    mindspore.nn.MatDet
    mindspore.nn.MatInverse
    mindspore.nn.MatMul
    mindspore.nn.Moments
    mindspore.nn.ReduceLogSumExp

Metrics
--------

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Accuracy
    mindspore.nn.F1
    mindspore.nn.Fbeta
    mindspore.nn.get_metric_fn
    mindspore.nn.Loss
    mindspore.nn.MAE 
    mindspore.nn.Metric
    mindspore.nn.MSE
    mindspore.nn.names
    mindspore.nn.Precision
    mindspore.nn.Recall
    mindspore.nn.Top1CategoricalAccuracy
    mindspore.nn.Top5CategoricalAccuracy
    mindspore.nn.TopKCategoricalAccuracy

Dynamic Learning Rate
-----------------------

LearningRateSchedule
^^^^^^^^^^^^^^^^^^^^^^

The dynamic learning rates in this module are all subclasses of LearningRateSchedule. Pass the instance of
LearningRateSchedule to an optimizer. During the training process, the optimizer calls the instance taking current step
as input to get the current learning rate.

.. code-block:: python

    import mindspore.nn as nn

    min_lr = 0.01
    max_lr = 0.1
    decay_steps = 4
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)

    net = Net()
    optim = nn.Momentum(net.trainable_params(), learning_rate=cosine_decay_lr, momentum=0.9)

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CosineDecayLR
    mindspore.nn.ExponentialDecayLR
    mindspore.nn.InverseDecayLR
    mindspore.nn.NaturalExpDecayLR
    mindspore.nn.PolynomialDecayLR
    mindspore.nn.WarmUpLR

Dynamic LR
^^^^^^^^^^^^^^^^^^^^^^

The dynamic learning rates in this module are all functions. Call the function and pass the result to an optimizer.
During the training process, the optimizer takes result[current step] as current learning rate.

.. code-block:: python

    import mindspore.nn as nn

    min_lr = 0.01
    max_lr = 0.1
    total_step = 6
    step_per_epoch = 1
    decay_epoch = 4

    lr= nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)

    net = Net()
    optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

.. autosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.cosine_decay_lr
    mindspore.nn.exponential_decay_lr
    mindspore.nn.inverse_decay_lr
    mindspore.nn.natural_exp_decay_lr
    mindspore.nn.piecewise_constant_lr
    mindspore.nn.polynomial_decay_lr
    mindspore.nn.warmup_lr
