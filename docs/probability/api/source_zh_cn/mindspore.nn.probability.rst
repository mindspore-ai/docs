mindspore.nn.probability
========================

.. automodule:: mindspore.nn.probability

Bijectors
---------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.bijector.Bijector
    mindspore.nn.probability.bijector.Exp
    mindspore.nn.probability.bijector.GumbelCDF
    mindspore.nn.probability.bijector.Invert
    mindspore.nn.probability.bijector.PowerTransform
    mindspore.nn.probability.bijector.ScalarAffine
    mindspore.nn.probability.bijector.Softplus

Bayesian Layers
---------------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.bnn_layers.ConvReparam
    mindspore.nn.probability.bnn_layers.DenseLocalReparam
    mindspore.nn.probability.bnn_layers.DenseReparam

Prior and Posterior Distributions
----------------------------------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.bnn_layers.NormalPosterior
    mindspore.nn.probability.bnn_layers.NormalPrior

Bayesian Wrapper Functions
---------------------------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.bnn_layers.WithBNNLossCell

Distributions
--------------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.distribution.Bernoulli
    mindspore.nn.probability.distribution.Beta
    mindspore.nn.probability.distribution.Categorical
    mindspore.nn.probability.distribution.Cauchy
    mindspore.nn.probability.distribution.Distribution
    mindspore.nn.probability.distribution.Exponential
    mindspore.nn.probability.distribution.Gamma
    mindspore.nn.probability.distribution.Geometric
    mindspore.nn.probability.distribution.Gumbel
    mindspore.nn.probability.distribution.Logistic
    mindspore.nn.probability.distribution.LogNormal
    mindspore.nn.probability.distribution.Normal
    mindspore.nn.probability.distribution.Poisson
    mindspore.nn.probability.distribution.TransformedDistribution
    mindspore.nn.probability.distribution.Uniform

Deep Probability Networks
--------------------------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.dpn.ConditionalVAE
    mindspore.nn.probability.dpn.VAE

Infer
------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.infer.ELBO
    mindspore.nn.probability.infer.SVI

ToolBox
---------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.toolbox.UncertaintyEvaluation
    mindspore.nn.probability.toolbox.VAEAnomalyDetection

Model Transformer
------------------

.. msplatformautosummary::
    :toctree: nn_probability
    :nosignatures:
    :template: classtemplate_probability.rst

    mindspore.nn.probability.transforms.TransformToBNN
