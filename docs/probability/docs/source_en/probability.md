# Deep Probabilistic Programming Library

<a href="https://gitee.com/mindspore/docs/blob/master/docs/probability/docs/source_en/probability.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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
import mindspore as ms
import mindspore.nn.probability.distribution as msd
ms.set_context(mode=ms.PYNATIVE_MODE)

my_normal = msd.Normal(0.0, 1.0, dtype=ms.float32)

mean = my_normal.mean()
var = my_normal.var()
entropy = my_normal.entropy()

value = ms.Tensor([-0.5, 0.0, 0.5], dtype=ms.float32)
prob = my_normal.prob(value)
cdf = my_normal.cdf(value)

mean_b = ms.Tensor(1.0, dtype=ms.float32)
sd_b = ms.Tensor(2.0, dtype=ms.float32)
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
import mindspore as ms
import mindspore.nn.probability.distribution as msd
ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.normal = msd.Normal(0.0, 1.0, dtype=ms.float32)

    def construct(self, value, mean, sd):
        pdf = self.normal.prob(value)
        kl = self.normal.kl_loss("Normal", mean, sd)
        return pdf, kl

net = Net()
value = ms.Tensor([-0.5, 0.0, 0.5], dtype=ms.float32)
mean = ms.Tensor(1.0, dtype=ms.float32)
sd = ms.Tensor(1.0, dtype=ms.float32)
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
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)

normal = msd.Normal(0.0, 1.0, dtype=ms.float32)
exp = msb.Exp()
LogNormal = msd.TransformedDistribution(exp, normal, seed=0, name="LogNormal")

# compute cumulative distribution function
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = ms.Tensor(x, dtype=ms.float32)
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
normal = msd.Normal(0.0, 1.0, dtype=ms.float32)
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
import mindspore as ms
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self, shape, dtype=ms.float32, seed=0, name='transformed_distribution'):
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
tx = ms.Tensor(x, dtype=ms.float32)
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
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)

powertransform = msb.PowerTransform(power=2.)

x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
tx = ms.Tensor(x, dtype=ms.float32)
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
import mindspore as ms
import mindspore.nn.probability.bijector as msb
ms.set_context(mode=ms.GRAPH_MODE)

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
tx = ms.Tensor(x, dtype=ms.float32)
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
