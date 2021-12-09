# Deep Probabilistic Programming Library

<!-- TOC -->

- [Deep Probabilistic Programming Library](#deep-probabilistic-programming-library)
    - [Probability Distribution](#probability-distribution)
        - [Probability Distribution Class](#probability-distribution-class)
            - [Distribution Base Class](#distribution-base-class)
            - [Bernoulli Distribution](#bernoulli-distribution)
            - [Exponential Distribution](#exponential-distribution)
            - [Geometric Distribution](#geometric-distribution)
            - [Normal Distribution](#normal-distribution)
            - [Uniform Distribution](#uniform-distribution)
            - [Categorical Distribution](#categorical-distribution)
            - [Cauchy Distribution](#cauchy-distribution)
            - [LogNormal Distribution](#lognormal-distribution)
            - [Gumbel Distribution](#gumbel-distribution)
            - [Logistic Distribution](#logistic-distribution)
            - [Poisson Distribution](#poisson-distribution)
            - [Gamma Distribution](#gamma-distribution)
            - [Beta Distribution](#beta-distribution)
        - [Probability Distribution Class Application in PyNative Mode](#probability-distribution-class-application-in-pynative-mode)
        - [Probability Distribution Class Application in Graph Mode](#probability-distribution-class-application-in-graph-mode)
        - [TransformedDistribution Class API Design](#transformeddistribution-class-api-design)
        - [Invoking a TransformedDistribution Instance in PyNative Mode](#invoking-a-transformeddistribution-instance-in-pynative-mode)
        - [Invoking a TransformedDistribution Instance in Graph Mode](#invoking-a-transformeddistribution-instance-in-graph-mode)
    - [Probability Distribution Mapping](#probability-distribution-mapping)
        - [Bijector API Design](#bijector-api-design)
            - [Bijector Base Class](#bijector-base-class)
            - [PowerTransform](#powertransform)
            - [Exp](#exp)
            - [ScalarAffine](#scalaraffine)
            - [Softplus](#softplus)
            - [GumbelCDF](#gumbelcdf)
            - [Invert](#invert)
        - [Invoking the Bijector Instance in PyNative Mode](#invoking-the-bijector-instance-in-pynative-mode)
        - [Invoking a Bijector Instance in Graph Mode](#invoking-a-bijector-instance-in-graph-mode)
    - [Deep Probabilistic Network](#deep-probabilistic-network)
        - [VAE](#vae)
        - [ConditionalVAE](#conditionalvae)
    - [Probability Inference Algorithm](#probability-inference-algorithm)
    - [Bayesian Layer](#bayesian-layer)
    - [Bayesian Conversion](#bayesian-conversion)
    - [Bayesian Toolbox](#bayesian-toolbox)
        - [Uncertainty Estimation](#uncertainty-estimation)
        - [Anomaly Detection](#anomaly-detection)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/probability/docs/source_en/probability.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

MindSpore deep probabilistic programming is to combine Bayesian learning with deep learning, including probability distribution, probability distribution mapping, deep probability network, probability inference algorithm, Bayesian layer, Bayesian conversion, and Bayesian toolkit. For professional Bayesian learning users, it provides probability sampling, inference algorithms, and model build libraries. On the other hand, advanced APIs are provided for users who are unfamiliar with Bayesian deep learning, so that they can use Bayesian models without changing the deep learning programming logic.

## Probability Distribution

Probability distribution (`mindspore.nn.probability.distribution`) is the basis of probabilistic programming. The `Distribution` class provides various probability statistics APIs, such as *pdf* for probability density, *cdf* for cumulative density, *kl_loss* for divergence calculation, and *sample* for sampling. Existing probability distribution examples include Gaussian distribution, Bernoulli distribution, exponential distribution, geometric distribution, and uniform distribution.

### Probability Distribution Class

- `Distribution`: base class of all probability distributions.

- `Bernoulli`: Bernoulli distribution, with a parameter indicating the number of experiment successes.

- `Exponential`: exponential distribution, with a rate parameter.

- `Geometric`: geometric distribution, with a parameter indicating the probability of initial experiment success.

- `Normal`: normal distribution (Gaussian distribution), with two parameters indicating the average value and standard deviation.

- `Uniform`: uniform distribution, with two parameters indicating the minimum and maximum values on the axis.

- `Categorical`: categorical distribution, with one parameter indicating the probability of each category.

- `Cauchy`: cauchy distribution, with two parameters indicating the location and scale.

- `LogNormal`: lognormal distribution, with two parameters indicating the location and scale.

- `Logistic`: logistic distribution, with two parameters indicating the location and scale.

- `Gumbel`: gumbel distribution, with two parameters indicating the location and scale.

#### Distribution Base Class

`Distribution` is the base class for all probability distributions.

The `Distribution` class supports the following functions: `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`, `mean`, `sd`, `var`, `entropy`, `kl_loss`, `cross_entropy`, and `sample`. The input parameters vary according to the distribution. These functions can be used only in a derived class and their parameters are determined by the function implementation of the derived class.

- `prob`: probability density function (PDF) or probability quality function (PMF)
- `log_prob`: log-like function
- `cdf`: cumulative distribution function (CDF)
- `log_cdf`: log-cumulative distribution function
- `survival_function`: survival function
- `log_survival`: logarithmic survival function
- `mean`: average value
- `sd`: standard deviation
- `var`: variance
- `entropy`: entropy
- `kl_loss`: Kullback-Leibler divergence
- `cross_entropy`: cross entropy of two probability distributions
- `sample`: random sampling of probability distribution
- `get_dist_args`: returns the parameters of the distribution used in the network
- `get_dist_name`: returns the type of the distribution

#### Bernoulli Distribution

Bernoulli distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Bernoulli.probs`: returns the probability of success in the Bernoulli experiment as a `Tensor`.

The `Distribution` base class invokes the private API in the `Bernoulli` to implement the public APIs in the base class. `Bernoulli` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input parameter *probs1* that indicates the probability of experiment success is optional.
- `entropy`: The input parameter *probs1* that indicates the probability of experiment success is optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist* and *probs1_b* are mandatory. *dist* indicates another distribution type. Currently, only *'Bernoulli'* is supported. *probs1_b* is the experiment success probability of distribution *b*. Parameter *probs1_a* of distribution *a* is optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. The input parameter *probs* that indicates the probability of experiment success is optional.
- `sample`: Optional input parameters include sample shape *shape* and experiment success probability *probs1*.
- `get_dist_args`: The input parameter *probs1* that indicates the probability of experiment success is optional. Return `(probs1,)` with type tuple.
- `get_dist_type`: return *'Bernoulli'*.

#### Exponential Distribution

Exponential distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Exponential.rate`: returns the rate parameter as a `Tensor`.

The `Distribution` base class invokes the `Exponential` private API to implement the public APIs in the base class. `Exponential` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input rate parameter *rate* is optional.
- `entropy`: The input rate parameter *rate* is optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist* and *rate_b* are mandatory.  *dist* indicates the name of another distribution type. Currently, only *'Exponential'* is supported. *rate_b* is the rate parameter of distribution *b*. Parameter *rate_a* of distribution *a* is optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. The input rate parameter *rate*is optional.
- `sample`: Optional input parameters include sample shape *shape* and rate parameter *rate*.
- `get_dist_args`: The input rate parameter *rate* is optional. Return `(rate,)` with type tuple.
- `get_dist_type`: returns *'Exponential'*.

#### Geometric Distribution

Geometric distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Geometric.probs`: returns the probability of success in the Bernoulli experiment as a `Tensor`.

The `Distribution` base class invokes the private API in the `Geometric` to implement the public APIs in the base class. `Geometric` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input parameter *probs1* that indicates the probability of experiment success is optional.
- `entropy`: The input parameter *probs1* that indicates the probability of experiment success is optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist* and *probs1_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Geometric'* is supported. *probs1_b* is the experiment success probability of distribution *b*. Parameter *probs1_a* of distribution *a* is optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. The input parameter *probs1* that indicates the probability of experiment success is optional.
- `sample`: Optional input parameters include sample shape *shape* and experiment success probability *probs1*.
- `get_dist_args`: The input parameter *probs1* that indicates the probability of experiment success is optional. Return `(probs1,)` with type tuple.
- `get_dist_type`: returns *'Geometric'*.

#### Normal Distribution

Normal distribution (also known as Gaussian distribution), inherited from the `Distribution` class.

The `Distribution` base class invokes the private API in the `Normal` to implement the public APIs in the base class. `Normal` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: Input parameters *mean* (for average value) and *sd* (for standard deviation) are optional.
- `entropy`: Input parameters *mean* (for average value) and *sd* (for standard deviation) are optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *mean_b*, and *sd_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Normal'* is supported. *mean_b* and *sd_b* indicate the mean value and standard deviation of distribution *b*, respectively. Input parameters mean value *mean_a* and standard deviation *sd_a* of distribution *a* are optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. Input parameters mean value *mean_a* and standard deviation *sd_a* are optional.
- `sample`: Input parameters sample shape *shape*, average value *mean_a*, and standard deviation *sd_a* are optional.
- `get_dist_args`: Input parameters mean value *mean* and standard deviation *sd* are optional. Return `(mean, sd)` with type tuple.
- `get_dist_type`: returns *'Normal'*.

#### Uniform Distribution

Uniform distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Uniform.low`: returns the minimum value as a `Tensor`.
- `Uniform.high`: returns the maximum value as a `Tensor`.

The `Distribution` base class invokes `Uniform` to implement public APIs in the base class. `Uniform` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: Input parameters maximum value *high* and minimum value *low* are optional.
- `entropy`: Input parameters maximum value *high* and minimum value *low* are optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *high_b*, and *low_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Uniform'* is supported. *high_b* and *low_b* are parameters of distribution *b*. Input parameters maximum value *high* and minimum value *low* of distribution *a* are optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. Input parameters maximum value *high* and minimum value *low* are optional.
- `sample`: Input parameters *shape*, maximum value *high*, and minimum value *low* are optional.
- `get_dist_args`: Input parameters maximum value *high* and minimum value *low* are optional. Return `(low, high)` with type tuple.
- `get_dist_type`: returns *'Uniform'*.

#### Categorical Distribution

Categorical distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Categorical.probs`: returns the probability of each category as a `Tensor`.

The `Distribution` base class invokes the private API in the `Categorical` to implement the public APIs in the base class. `Categorical` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input parameter *probs* that indicates the probability of each category is optional.
- `entropy`: The input parameter *probs* that indicates the probability of each category is optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist* and *probs_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Categorical'* is supported. *probs_b* is the categories' probabilities of distribution *b*. Parameter *probs_a* of distribution *a* is optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. The input parameter *probs* that indicates the probability of each category is optional.
- `sample`: Optional input parameters include sample shape *shape* and the categories' probabilities *probs*.
- `get_dist_args`: The input parameter *probs* that indicates the probability of each category is optional. Return `(probs,)` with type tuple.
- `get_dist_type`: returns *'Categorical'*.

#### Cauchy Distribution

Cauchy distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Cauchy.loc`: returns the location parameter as a `Tensor`.
- `Cauchy.scale`: returns the scale parameter as a `Tensor`.

The `Distribution` base class invokes the private API in the `Cauchy` to implement the public APIs in the base class. `Cauchy` supports the following public APIs:

- `entropy`: Input parameters *loc* (for location) and *scale* (for scale) are optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *loc_b*, and *scale_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Cauchy'* is supported. *loc_b* and *scale_b* indicate the location and scale of distribution *b*, respectively. Input parameters *loc* and *scale* of distribution *a* are optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. Input parameters location *loc* and scale *scale* are optional.
- `sample`: Input parameters sample shape *shape*, location *loc* and scale *scale* are optional.
- `get_dist_args`: Input parameters location *loc* and scale *scale* are optional. Return `(loc, scale)` with type tuple.
- `get_dist_type`: returns *'Cauchy'*.

#### LogNormal Distribution

LogNormal distribution, inherited from the `TransformedDistribution` class, constructed by `Exp` Bijector and `Normal` Distribution.

Properties are described as follows:

- `LogNormal.loc`: returns the location parameter as a `Tensor`.
- `LogNormal.scale`: returns the scale parameter as a `Tensor`.

The `Distribution` base class invokes the private API in the `LogNormal` and `TransformedDistribution` to implement the public APIs in the base class. `LogNormal` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`：Input parameters *loc* (for location) and *scale* (for scale) are optional.
- `entropy`: Input parameters *loc* (for location) and *scale* (for scale) are optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *loc_b*, and *scale_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'LogNormal'* is supported. *loc_b* and *scale_b* indicate the location and scale of distribution *b*, respectively. Input parameters *loc* and *scale* of distribution *a* are optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. Input parameters location *loc* and scale *scale* are optional.
- `sample`: Input parameters sample shape *shape*, location *loc* and scale *scale* are optional.
- `get_dist_args`: Input parameters location *loc* and scale *scale* are optional. Return `(loc, scale)` with type tuple.
- `get_dist_type`: returns *'LogNormal'*.

#### Gumbel Distribution

Gumbel distribution, inherited from the `TransformedDistribution` class, constructed by `GumbelCDF` Bijector and `Uniform` Distribution.

Properties are described as follows:

- `Gumbel.loc`: returns the location parameter as a `Tensor`.
- `Gumbel.scale`: returns the scale parameter as a `Tensor`.

The `Distribution` base class invokes the private API in the `Gumbel` and `TransformedDistribution` to implement the public APIs in the base class. `Gumbel` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`：No parameter.
- `entropy`: No parameter.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *loc_b*, and *scale_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Gumbel'* is supported. *loc_b* and *scale_b* indicate the location and scale of distribution *b*.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory.
- `sample`: Input parameters sample shape *shape* is optional.
- `get_dist_args`: Input parameters location *loc* and scale *scale* are optional. Return `(loc, scale)` with type tuple.
- `get_dist_type`: returns *'Gumbel'*.

#### Logistic Distribution

Logistic distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Logistic.loc`: returns the location parameter as a `Tensor`.
- `Logistic.scale`: returns the scale parameter as a `Tensor`.

The `Distribution` base class invokes the private API in the `Logistic` and `TransformedDistribution` to implement the public APIs in the base class. `Logistic` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`：Input parameters *loc* (for location) and *scale* (for scale) are optional.
- `entropy`: Input parameters *loc* (for location) and *scale* (for scale) are optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. Input parameters location *loc* and scale *scale* are optional.
- `sample`: Input parameters sample shape *shape*, location *loc* and scale *scale* are optional.
- `get_dist_args`: Input parameters location *loc* and scale *scale* are optional. Return `(loc, scale)` with type tuple.
- `get_dist_type`: returns *'Logistic'*.

#### Poisson Distribution

Poisson distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Poisson.rate`: returns the rate as a `Tensor`.

The `Distribution` base class invokes the private API in the `Poisson` to implement the public APIs in the base class. `Poisson` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input parameter *rate* is optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. The input parameter rate* is optional.
- `sample`: Optional input parameters include sample shape *shape* and the parameter *rate*.
- `get_dist_args`: The input parameter *rate* is optional. Return `(rate,)` with type tuple.
- `get_dist_type`: returns *'Poisson'*.

#### Gamma Distribution

Gamma distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Gamma.concentration`: returns the concentration as a `Tensor`.
- `Gamma.rate`: returns the rate as a `Tensor`.

The `Distribution` base class invokes the private API in the `Gamma` to implement the public APIs in the base class. `Gamma` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input parameters *concentration* and *rate* are optional.
- `entropy`: The input parameters *concentration* and *rate* are optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *concentration_b* and *rate_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Gamma'* is supported. *concentration_b* and *rate_b* are the parameters of distribution *b*. The input parameters *concentration_a* and *rate_a* for distribution *a* are optional.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`: The input parameter *value* is mandatory. The input parameters *concentration* and *rate* are optional.
- `sample`: Optional input parameters include sample shape *shape* and parameters *concentration* and *rate*.
- `get_dist_args`: The input parameters *concentration* and *rate* are optional. Return `(concentration, rate)` with type tuple.
- `get_dist_type`: returns *'Gamma'*.

#### Beta Distribution

Beta distribution, inherited from the `Distribution` class.

Properties are described as follows:

- `Beta.concentration1`: returns the rate as a `Tensor`.
- `Beta.concentration0`: returns the rate as a `Tensor`.

The `Distribution` base class invokes the private API in the `Beta` to implement the public APIs in the base class. `Beta` supports the following public APIs:

- `mean`,`mode`,`var`, and `sd`: The input parameters *concentration1* and *concentration0* are optional.
- `entropy`: The input parameters *concentration1* and *concentration0* are optional.
- `cross_entropy` and `kl_loss`: The input parameters *dist*, *concentration1_b* and *rateconcentration0_b* are mandatory. *dist* indicates the name of another distribution type. Currently, only *'Beta'* is supported. *concentration1_b* and *concentration0_b* are the parameters of distribution *b*. The input parameters *concentratio1n_a* and *concentration0_a* for distribution *a* are optional.
- `prob` and `log_prob`: The input parameter *value* is mandatory. The input parameters *concentration1* and *concentration0* are optional.
- `sample`: Optional input parameters include sample shape *shape* and parameters *concentration1* and *concentration0*.
- `get_dist_args`: The input parameters *concentration1* and *concentration0* are optional. Return `(concentration1, concentration0)` with type tuple.
- `get_dist_type`: returns *'Beta'*.

### Probability Distribution Class Application in PyNative Mode

`Distribution` subclasses can be used in **PyNative** mode.

Use `Normal` as an example. Create a normal distribution whose average value is 0.0 and standard deviation is 1.0.

```python
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.PYNATIVE_MODE)

my_normal = msd.Normal(0.0, 1.0, dtype=mstype.float32)

mean = my_normal.mean()
var = my_normal.var()
entropy = my_normal.entropy()

value = Tensor([-0.5, 0.0, 0.5], dtype=mstype.float32)
prob = my_normal.prob(value)
cdf = my_normal.cdf(value)

mean_b = Tensor(1.0, dtype=mstype.float32)
sd_b = Tensor(2.0, dtype=mstype.float32)
kl = my_normal.kl_loss('Normal', mean_b, sd_b)

# get the distribution args as a tuple
dist_arg = my_normal.get_dist_args()

print("mean: ", mean)
print("var: ", var)
print("entropy: ", entropy)
print("prob: ", prob)
print("cdf: ", cdf)
print("kl: ", kl)
print("dist_arg: ", dist_arg)
```

The output is as follows:

```text
mean:  0.0
var:  1.0
entropy:  1.4189385
prob:  [0.35206532 0.3989423  0.35206532]
cdf:  [0.30853754 0.5        0.69146246]
kl:  0.44314718
dist_arg: (Tensor(shape=[], dtype=Float32, value= 0), Tensor(shape=[], dtype=Float32, value= 1))
```

### Probability Distribution Class Application in Graph Mode

In graph mode, `Distribution` subclasses can be used on the network.

```python
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.normal = msd.Normal(0.0, 1.0, dtype=mstype.float32)

    def construct(self, value, mean, sd):
        pdf = self.normal.prob(value)
        kl = self.normal.kl_loss("Normal", mean, sd)
        return pdf, kl

net = Net()
value = Tensor([-0.5, 0.0, 0.5], dtype=mstype.float32)
mean = Tensor(1.0, dtype=mstype.float32)
sd = Tensor(1.0, dtype=mstype.float32)
pdf, kl = net(value, mean, sd)
print("pdf: ", pdf)
print("kl: ", kl)
```

The output is as follows:

```text
pdf:  [0.35206532 0.3989423  0.35206532]
kl:  0.5
```

### TransformedDistribution Class API Design

`TransformedDistribution`, inherited from `Distribution`, is a base class for mathematical distribution that can be obtained by mapping f(x) changes. The APIs are as follows:

1. Properties

    - `bijector`: returns the distribution transformation method.
    - `distribution`: returns the original distribution.
    - `is_linear_transformation`: returns the linear transformation flag.

2. API functions (The parameters of the following APIs are the same as those of the corresponding APIs of `distribution` in the constructor function.)

    - `cdf`: cumulative distribution function (CDF)
    - `log_cdf`: log-cumulative distribution function
    - `survival_function`: survival function
    - `log_survival`: logarithmic survival function
    - `prob`: probability density function (PDF) or probability quality function (PMF)
    - `log_prob`: log-like function
    - `sample`: random sampling
    - `mean`: a non-parametric function, which can be invoked only when `Bijector.is_constant_jacobian=true` is invoked.

### Invoking a TransformedDistribution Instance in PyNative Mode

The `TransformedDistribution` subclass can be used in **PyNative** mode.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
import mindspore.context as context
from mindspore import Tensor, dtype

context.set_context(mode=context.PYNATIVE_MODE)

normal = msd.Normal(0.0, 1.0, dtype=dtype.float32)
exp = msb.Exp()
LogNormal = msd.TransformedDistribution(exp, normal, seed=0, name="LogNormal")

# compute cumulative distribution function
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
cdf = LogNormal.cdf(tx)

# generate samples from the distribution
shape = (3, 2)
sample = LogNormal.sample(shape)

# get information of the distribution
print(LogNormal)
# get information of the underlying distribution and the bijector separately
print("underlying distribution:\n", LogNormal.distribution)
print("bijector:\n", LogNormal.bijector)
# get the computation results
print("cdf:\n", cdf)
print("sample shape:\n", sample.shape)
```

The output is as follows:

```text
TransformedDistribution<
  (_bijector): Exp<exp>
  (_distribution): Normal<mean = 0.0, standard deviation = 1.0>
  >
underlying distribution:
 Normal<mean = 0.0, standard deviation = 1.0>
bijector:
 Exp<exp>
cdf:
 [0.7558914 0.9462397 0.9893489]
sample shape:
(3, 2)
```

When the `TransformedDistribution` is constructed to map the transformed `is_constant_jacobian = true` (for example, `ScalarAffine`), the constructed `TransformedDistribution` instance can use the `mean` API to calculate the average value. For example:

```python
normal = msd.Normal(0.0, 1.0, dtype=dtype.float32)
scalaraffine = msb.ScalarAffine(1.0, 2.0)
trans_dist = msd.TransformedDistribution(scalaraffine, normal, seed=0)
mean = trans_dist.mean()
print(mean)
```

The output is as follows:

```text
2.0
```

### Invoking a TransformedDistribution Instance in Graph Mode

In graph mode, the `TransformedDistribution` class can be used on the network.

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, dtype
import mindspore.context as context
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self, shape, dtype=dtype.float32, seed=0, name='transformed_distribution'):
        super(Net, self).__init__()
        # create TransformedDistribution distribution
        self.exp = msb.Exp()
        self.normal = msd.Normal(0.0, 1.0, dtype=dtype)
        self.lognormal = msd.TransformedDistribution(self.exp, self.normal, seed=seed, name=name)
        self.shape = shape

    def construct(self, value):
        cdf = self.lognormal.cdf(value)
        sample = self.lognormal.sample(self.shape)
        return cdf, sample

shape = (2, 3)
net = Net(shape=shape, name="LogNormal")
x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
tx = Tensor(x, dtype=dtype.float32)
cdf, sample = net(tx)
print("cdf: ", cdf)
print("sample shape: ", sample.shape)
```

The output is as follows:

```text
cdf:  [0.7558914  0.86403143 0.9171715  0.9462397 ]
sample shape:  (2, 3)
```

## Probability Distribution Mapping

Bijector (`mindspore.nn.probability.bijector`) is a basic component of probability programming. Bijector describes a random variable transformation method, and a new random variable $Y = f(x)$ may be generated by using an existing random variable X and a mapping function f.
`Bijector` provides four mapping-related transformation methods. It can be directly used as an operator, or used to generate a `Distribution` class instance of a new random variable on an existing `Distribution` class instance.

### Bijector API Design

#### Bijector Base Class

The `Bijector` class is the base class for all probability distribution mappings. The APIs are as follows:

1. Properties
   - `name`: returns the value of `name`.
   - `dtype`: returns the value of `dtype`.
   - `parameters`: returns the value of `parameter`.
   - `is_constant_jacobian`: returns the value of `is_constant_jacobian`.
   - `is_injective`: returns the value of `is_injective`.

2. Mapping functions
   - `forward`: forward mapping, whose parameter is determined by `_forward` of the derived class.
   - `inverse`: backward mapping, whose parameter is determined by`_inverse` of the derived class.
   - `forward_log_jacobian`: logarithm of the derivative of the forward mapping, whose parameter is determined by `_forward_log_jacobian` of the derived class.
   - `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, whose parameter is determined by `_inverse_log_jacobian` of the derived class.

When `Bijector` is invoked as a function:
The input is a `Distribution` class and a `TransformedDistribution` is generated **(cannot be invoked in a graph)**.

#### PowerTransform

`PowerTransform` implements variable transformation with $Y = g(X) = {(1 + X * c)}^{1 / c}$. The APIs are as follows:

1. Properties
   - `power`: returns the value of `power` as a `Tensor`.

2. Mapping functions
   - `forward`: forward mapping, with an input parameter `Tensor`.
   - `inverse`: backward mapping, with an input parameter `Tensor`.
   - `forward_log_jacobian`: logarithm of the derivative of the forward mapping, with an input parameter `Tensor`.
   - `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, with an input parameter `Tensor`.

#### Exp

`Exp` implements variable transformation with $Y = g(X)= exp(X)$. The APIs are as follows:

Mapping functions

- `forward`: forward mapping, with an input parameter `Tensor`.
- `inverse`: backward mapping, with an input parameter `Tensor`.
- `forward_log_jacobian`: logarithm of the derivative of the forward mapping, with an input parameter `Tensor`.
- `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, with an input parameter `Tensor`.

#### ScalarAffine

`ScalarAffine` implements variable transformation with Y = g(X) = a * X + b. The APIs are as follows:

1. Properties
    - `scale`: returns the value of scale as a `Tensor`.
    - `shift`: returns the value of shift as a `Tensor`.

2. Mapping functions
    - `forward`: forward mapping, with an input parameter `Tensor`.
    - `inverse`: backward mapping, with an input parameter `Tensor`.
    - `forward_log_jacobian`: logarithm of the derivative of the forward mapping, with an input parameter `Tensor`.
    - `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, with an input parameter `Tensor`.

#### Softplus

`Softplus` implements variable transformation with $Y = g(X) = log(1 + e ^ {kX}) / k $. The APIs are as follows:

1. Properties
    - `sharpness`: returns the value of `sharpness` as a `Tensor`.

2. Mapping functions
    - `forward`: forward mapping, with an input parameter `Tensor`.
    - `inverse`: backward mapping, with an input parameter `Tensor`.
    - `forward_log_jacobian`: logarithm of the derivative of the forward mapping, with an input parameter `Tensor`.
    - `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, with an input parameter `Tensor`.

#### GumbelCDF

`GumbelCDF` implements variable transformation with $Y = g(X) = \exp(-\exp(-\frac{X - loc}{scale}))$. The APIs are as follows:

1. Properties
    - `loc`: returns the value of `loc` as a `Tensor`.
    - `scale`: returns the value of `scale` as a `Tensor`.

2. Mapping functions
    - `forward`: forward mapping, with an input parameter `Tensor`.
    - `inverse`: backward mapping, with an input parameter `Tensor`.
    - `forward_log_jacobian`: logarithm of the derivative of the forward mapping, with an input parameter `Tensor`.
    - `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, with an input parameter `Tensor`.

#### Invert

`Invert` implements the inverse of another bijector. The APIs are as follows:

1. Properties
    - `bijector`: returns the Bijector used during initialization with type `msb.Bijector`.

2. Mapping functions
    - `forward`: forward mapping, with an input parameter `Tensor`.
    - `inverse`: backward mapping, with an input parameter `Tensor`.
    - `forward_log_jacobian`: logarithm of the derivative of the forward mapping, with an input parameter `Tensor`.
    - `inverse_log_jacobian`: logarithm of the derivative of the backward mapping, with an input parameter `Tensor`.

### Invoking the Bijector Instance in PyNative Mode

Before the execution, import the required library file package. The main library of the Bijector class is `mindspore.nn.probability.bijector`. After the library is imported, `msb` is used as the abbreviation of the library for invoking.

The following uses `PowerTransform` as an example. Create a `PowerTransform` object whose power is 2.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.context as context
from mindspore import Tensor, dtype

context.set_context(mode=context.PYNATIVE_MODE)

powertransform = msb.PowerTransform(power=2.)

x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
forward = powertransform.forward(tx)
inverse = powertransform.inverse(tx)
forward_log_jaco = powertransform.forward_log_jacobian(tx)
inverse_log_jaco = powertransform.inverse_log_jacobian(tx)

print(powertransform)
print("forward: ", forward)
print("inverse: ", inverse)
print("forward_log_jacobian: ", forward_log_jaco)
print("inverse_log_jacobian: ", inverse_log_jaco)
```

The output is as follows:

```text
PowerTransform<power = 2.0>
forward:  [2.236068  2.6457515 3.        3.3166249]
inverse:  [ 1.5       4.        7.5      12.000001]
forward_log_jacobian:  [-0.804719  -0.9729551 -1.0986123 -1.1989477]
inverse_log_jacobian:  [0.6931472 1.0986123 1.3862944 1.609438 ]
```

### Invoking a Bijector Instance in Graph Mode

In graph mode, the `Bijector` subclass can be used on the network.

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.bijector as msb
context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        # create a PowerTransform bijector
        self.powertransform = msb.PowerTransform(power=2.)

    def construct(self, value):
        forward = self.powertransform.forward(value)
        inverse = self.powertransform.inverse(value)
        forward_log_jaco = self.powertransform.forward_log_jacobian(value)
        inverse_log_jaco = self.powertransform.inverse_log_jacobian(value)
        return forward, inverse, forward_log_jaco, inverse_log_jaco

net = Net()
x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
tx = Tensor(x, dtype=mstype.float32)
forward, inverse, forward_log_jaco, inverse_log_jaco = net(tx)
print("forward: ", forward)
print("inverse: ", inverse)
print("forward_log_jaco: ", forward_log_jaco)
print("inverse_log_jaco: ", inverse_log_jaco)
```

The output is as follows:

```text
forward:  [2.236068  2.6457515 3.        3.3166249]
inverse:  [ 1.5       4.        7.5      12.000001]
forward_log_jacobian:  [-0.804719  -0.9729551 -1.0986123 -1.1989477]
inverse_log_jacobian:  [0.6931472 1.0986123 1.3862944 1.609438 ]
```

## Deep Probabilistic Network

It is especially easy to use the MindSpore deep probabilistic programming library (`mindspore.nn.probability.dpn`) to construct a variational auto-encoder (VAE) for inference. You only need to define the encoder and decoder (a DNN model), invoke the VAE or conditional VAE (CVAE) API to form a derived network, invoke the ELBO API for optimization, and use the SVI API for variational inference. The advantage is that users who are not familiar with variational inference can build a probability model in the same way as they build a DNN model, and those who are familiar with variational inference can invoke these APIs to build a more complex probability model. VAE APIs are defined in `mindspore.nn.probability.dpn`, where dpn represents the deep probabilistic network. `mindspore.nn.probability.dpn` provides some basic APIs of the deep probabilistic network, for example, VAE.

### VAE

First, we need to define the encoder and decoder and invoke the `mindspore.nn.probability.dpn.VAE` API to construct the VAE network. In addition to the encoder and decoder, we need to input the hidden size of the encoder output variable and the latent size of the VAE network storage potential variable. Generally, the latent size is less than the hidden size.

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.probability.dpn import VAE

IMAGE_SHAPE = (-1, 1, 32, 32)


class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024, 800)
        self.fc2 = nn.Dense(800, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder()
decoder = Decoder()
vae = VAE(encoder, decoder, hidden_size=400, latent_size=20)
```

### ConditionalVAE

Similarly, the usage of CVAE is similar to that of VAE. The difference is that CVAE uses the label information of datasets. It is a supervised learning algorithm, and has a better generation effect than VAE.

First, define the encoder and decoder and invoke the `mindspore.nn.probability.dpn.ConditionalVAE` API to construct the CVAE network. The encoder here is different from that of the VAE because the label information of datasets needs to be input. The decoder is the same as that of the VAE. For the CVAE API, the number of dataset label categories also needs to be input. Other input parameters are the same as those of the VAE API.

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.probability.dpn import ConditionalVAE

IMAGE_SHAPE = (-1, 1, 32, 32)


class Encoder(nn.Cell):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024 + num_classes, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.concat = ops.Concat(axis=1)
        self.one_hot = nn.OneHot(depth=num_classes)

    def construct(self, x, y):
        x = self.flatten(x)
        y = self.one_hot(y)
        input_x = self.concat((x, y))
        input_x = self.fc1(input_x)
        input_x = self.relu(input_x)
        return input_x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder(num_classes=10)
decoder = Decoder()
cvae = ConditionalVAE(encoder, decoder, hidden_size=400, latent_size=20, num_classes=10)
```

Load a dataset, for example, Mnist. For details about the data loading and preprocessing process, see [Quick Start for Beginners](https://www.mindspore.cn/tutorials/en/master/quick_start.html). The create_dataset function is used to create a data iterator.

```python
ds_train = create_dataset(image_path, 128, 1)
```

Next, use the infer API to perform variational inference on the VAE network.

## Probability Inference Algorithm

Invoke the `mindspore.nn.probability.infer.ELBO` API to define the loss function of the VAE network, invoke `WithLossCell` to encapsulate the VAE network and loss function, define the optimizer, and transfer them to the `mindspore.nn.probability.infer.SVI` API. The `run` function of the SVI API can be understood to trigger training of the VAE network. You can specify the `epochs` of the training, so that a trained network is returned. If you specify the `get_train_loss` function, the loss of the trained model will be returned.

```python
from mindspore.nn.probability.infer import ELBO, SVI

net_loss = ELBO(latent_prior='Normal', output_prior='Normal')
net_with_loss = nn.WithLossCell(vae, net_loss)
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)

vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
vae = vi.run(train_dataset=ds_train, epochs=10)
trained_loss = vi.get_train_loss()
```

After obtaining the trained VAE network, use `vae.generate_sample` to generate a new sample. You need to specify the number of samples to be generated and the shape of the generated samples. The shape must be the same as that of the samples in the original dataset. You can also use `vae.reconstruct_sample` to reconstruct samples in the original dataset to test the reconstruction capability of the VAE network.

```python
generated_sample = vae.generate_sample(64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    reconstructed_sample = vae.reconstruct_sample(sample_x)
print('The shape of the generated sample is ', generated_sample.shape)
```

The shape of the newly generated sample is as follows:

```text
The shape of the generated sample is (64, 1, 32, 32)
```

The CVAE training process is similar to the VAE training process. However, when a trained CVAE network is used to generate a new sample and rebuild a new sample, label information needs to be input. For example, the generated new sample is 64 digits ranging from 0 to 7.

```python
sample_label = Tensor([i for i in range(0, 8)] * 8, dtype=mstype.int32)
generated_sample = cvae.generate_sample(sample_label, 64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    sample_y = Tensor(sample['label'], dtype=mstype.int32)
    reconstructed_sample = cvae.reconstruct_sample(sample_x, sample_y)
print('The shape of the generated sample is ', generated_sample.shape)
```

Check the shape of the newly generated sample:

```text
The shape of the generated sample is  (64, 1, 32, 32)
```

If you want the generated sample to be better and clearer, you can define a more complex encoder and decoder. The example uses only two layers of full-connected layers.

## Bayesian Layer

The following uses the APIs in `nn.probability.bnn_layers` of MindSpore to implement the BNN image classification model. The APIs in `nn.probability.bnn_layers` of MindSpore include `NormalPrior`, `NormalPosterior`, `ConvReparam`, `DenseReparam`, `DenseLocalReparam` and `WithBNNLossCell`. The biggest difference between BNN and DNN is that the weight and bias of the BNN layer are not fixed values, but follow a distribution. `NormalPrior` and `NormalPosterior` are respectively used to generate a prior distribution and a posterior distribution that follow a normal distribution. `ConvReparam` and `DenseReparam` are the Bayesian convolutional layer and fully connected layers implemented by using the reparameterization method, respectively. `DenseLocalReparam` is the Bayesian fully connected layers implemented by using the local reparameterization method. `WithBNNLossCell` is used to encapsulate the BNN and loss function.

For details about how to use the APIs in `nn.probability.bnn_layers` to build a Bayesian neural network and classify images, see [Applying the Bayesian Network](https://www.mindspore.cn/probability/docs/zh-CN/master/using_bnn.html).

## Bayesian Conversion

For researchers who are unfamiliar with the Bayesian model, the MDP provides the `mindspore.nn.probability.transform` API to convert the DNN model into the BNN model by one click.

The `__init__` function of the model conversion API `TransformToBNN` is defined as follows:

```python
class TransformToBNN:
    def __init__(self, trainable_dnn, dnn_factor=1, bnn_factor=1):
        net_with_loss = trainable_dnn.network
        self.optimizer = trainable_dnn.optimizer
        self.backbone = net_with_loss.backbone_network
        self.loss_fn = getattr(net_with_loss, "_loss_fn")
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.bnn_loss_file = None
```

The `trainable_bnn` parameter is a trainable DNN model packaged by `TrainOneStepCell`, `dnn_factor` and `bnn_factor` are the coefficient of the overall network loss calculated by the loss function and the coefficient of the KL divergence of each Bayesian layer, respectively.
`TransformToBNN` implements the following functions:

- Function 1: Convert the entire model.

  The `transform_to_bnn_model` method can convert the entire DNN model into a BNN model. The definition is as follows:

  ```python
    def transform_to_bnn_model(self,
                               get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias,
                                                          "out_channels": dp.out_channels, "activation": dp.activation},
                               get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels,
                                                         "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size,
                                                         "stride": dp.stride, "has_bias": dp.has_bias,
                                                         "padding": dp.padding, "dilation": dp.dilation,
                                                         "group": dp.group},
                               add_dense_args=None,
                               add_conv_args=None):
        r"""
        Transform the whole DNN model to BNN model, and wrap BNN model by TrainOneStepCell.

        Args:
            get_dense_args (function): The arguments gotten from the DNN full connection layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "has_bias": dp.has_bias}.
            get_conv_args (function): The arguments gotten from the DNN convolutional layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode,
                "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias}.
            add_dense_args (dict): The new arguments added to BNN full connection layer. Default: {}.
            add_conv_args (dict): The new arguments added to BNN convolutional layer. Default: {}.

        Returns:
            Cell, a trainable BNN model wrapped by TrainOneStepCell.
       """

  ```

  `get_dense_args` specifies the parameters to be obtained from the fully connected layer of the DNN model. The default value is the common parameters of the fully connected layers of the DNN and BNN models. For details about the parameters, see [mindspore API](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Dense.html). `get_conv_args` specifies the parameters to be obtained from the convolutional layer of the DNN model. The default value is the common parameters of the convolutional layers of the DNN and BNN models. For details about the parameters, see [MindSpore API](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Conv2d.html). `add_dense_args` and `add_conv_args` specify the new parameter values to be specified for the BNN layer. Note that the parameters in `add_dense_args` cannot be the same as those in `get_dense_args`. The same rule applies to `add_conv_args` and `get_conv_args`.

- Function 2: Convert a specific layer.

  The `transform_to_bnn_layer` method can convert a specific layer (`nn.Dense` or `nn.Conv2d`) in the DNN model into a corresponding Bayesian layer. The definition is as follows:

  ```python
   def transform_to_bnn_layer(self, dnn_layer, bnn_layer, get_args=None, add_args=None):
        r"""
        Transform a specific type of layers in DNN model to corresponding BNN layer.

        Args:
            dnn_layer_type (Cell): The type of DNN layer to be transformed to BNN layer. The optional values are
            nn.Dense, nn.Conv2d.
            bnn_layer_type (Cell): The type of BNN layer to be transformed to. The optional values are
                DenseReparameterization, ConvReparameterization.
            get_args (dict): The arguments gotten from the DNN layer. Default: None.
            add_args (dict): The new arguments added to BNN layer. Default: None.

        Returns:
            Cell, a trainable model wrapped by TrainOneStepCell, whose sprcific type of layer is transformed to the corresponding bayesian layer.
        """
  ```

  `Dnn_layer` specifies a DNN layer to be converted into a BNN layer, and `bnn_layer` specifies a BNN layer to be converted into a DNN layer, and `get_args` and `add_args` specify the parameters obtained from the DNN layer and the parameters to be re-assigned to the BNN layer, respectively.

For details about how to use `TransformToBNN` in MindSpore, see [DNN-to-BNN Conversion with One Click](https://www.mindspore.cn/probability/docs/zh-CN/master/one_click_conversion_from_dnn_to_bnn.html).

## Bayesian Toolbox

### Uncertainty Estimation

One of the advantages of the BNN is that uncertainty can be obtained. MDP provides a toolbox (`mindspore.nn.probability.toolbox`) for uncertainty estimation at the upper layer. You can easily use the toolbox to calculate uncertainty. Uncertainty means the uncertainty of the prediction result of the deep learning model. Currently, most deep learning algorithms can only provide high-confidence prediction results, but cannot determine the certainty of the prediction results. There are two types of uncertainty: aleatoric uncertainty and epistemic uncertainty.

- Aleatoric uncertainty: describes the internal noise of data, that is, the unavoidable error. This phenomenon cannot be weakened by adding sampling data.
- Epistemic uncertainty: describes the estimation inaccuracy of input data incurred due to reasons such as poor training or insufficient training data. This may be alleviated by adding training data.

The APIs of the uncertainty estimation toolbox are as follows:

- `model`: trained model whose uncertainty is to be estimated.
- `train_dataset`: dataset used for training, which is of the iterator type.
- `task_type`: model type. The value is a character string. Enter regression or classification.
- `num_classes`: For a classification model, you need to specify the number of labels of the classification.
- `epochs`: number of epochs for training an uncertain model.
- `epi_uncer_model_path`: path for storing or loading models that compute cognitive uncertainty.
- `ale_uncer_model_path`: path used to store or load models that calculate epistemic uncertainty.
- `save_model`: whether to store the model, which is of the Boolean type.

Before using the model, you need to train the model. The following uses LeNet5 as an example:

```python
from mindspore.nn.probability.toolbox import UncertaintyEvaluation
from mindspore import load_checkpoint, load_param_into_net

if __name__ == '__main__':
    # get trained model
    network = LeNet5()
    param_dict = load_checkpoint('checkpoint_lenet.ckpt')
    load_param_into_net(network, param_dict)
    # get train and eval dataset
    ds_train = create_dataset('workspace/mnist/train')
    ds_eval = create_dataset('workspace/mnist/test')
    evaluation = UncertaintyEvaluation(model=network,
                                       train_dataset=ds_train,
                                       task_type='classification',
                                       num_classes=10,
                                       epochs=1,
                                       epi_uncer_model_path=None,
                                       ale_uncer_model_path=None,
                                       save_model=False)
    for eval_data in ds_eval.create_dict_iterator():
        eval_data = Tensor(eval_data['image'], mstype.float32)
        epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
        aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
    print('The shape of epistemic uncertainty is ', epistemic_uncertainty.shape)
    print('The shape of epistemic uncertainty is ', aleatoric_uncertainty.shape)
```

`eval_epistemic_uncertainty` calculates epistemic uncertainty, which is also called model uncertainty. Each estimation label of every sample has an uncertain value. `eval_aleatoric_uncertainty` calculates aleatoric uncertainty, which is also called data uncertainty. Each sample has an uncertain value.
The output is as follows:

```text
The shape of epistemic uncertainty is (32, 10)
The shape of epistemic uncertainty is (32,)
```

The value of uncertainty is greater than or equal to zero. A larger value indicates higher uncertainty.

### Anomaly Detection

Anomaly Detection can find outliers that are "different from the main data distribution". For example, finding outliers in data preprocessing can help improve the model's fitting ability.

MDP provides anomaly detection toolbox (`VAEAnomalyDetection`) based on the variational autoencoder (VAE) in the upper layer. Similar to the use of VAE, we only need to customize the encoder and decoder (DNN model), initialize the relevant parameters, then you can use the toolbox to detect abnormal points.

The interface of the VAE-based anomaly detection toolbox is as follows:

- `encoder`：Encoder(Cell)
- `decoder`：Decoder(Cell)
- `hidden_size`：The size of encoder's output tensor
- `latent_size`：The size of the latent space

Use Encoder and Decoder, set hidden_size and latent_size, initialize the class, and then pass the dataset to detect abnormal points.

```python
from mindspore.nn.probability.toolbox import VAEAnomalyDetection

if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()
    ood = VAEAnomalyDetection(encoder=encoder, decoder=decoder,
                              hidden_size=400, latent_size=20)
    ds_train = create_dataset('workspace/mnist/train')
    ds_eval = create_dataset('workspace/mnist/test')
    model = ood.train(ds_train)
    for sample in ds_eval.create_dict_iterator(output_numpy=True, num_epochs=1):
        sample_x = Tensor(sample['image'], dtype=mstype.float32)
        score = ood.predict_outlier_score(sample_x)
        outlier = ood.predict_outlier(sample_x)
        print(score, outlier)
```

The output of `score` is the anomaly score of the sample. `outlier` is a Boolean type, `True` represents an abnormal point, and `False` represents a normal point.
