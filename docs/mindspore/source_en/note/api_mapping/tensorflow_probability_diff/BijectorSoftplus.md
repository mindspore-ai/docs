# Function Differences with tfp.bijectors.Softplus

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_probability_diff/BijectorSoftplus.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tfp.bijectors.Softplus

```python
class tfp.bijectors.Softplus(
    hinge_softness=None,
    low=None,
    validate_args=False,
    name='softplus'
)
```

For more information, see [tfp.bijectors.Softplus](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Softplus).

## mindspore.nn.probability.bijector.Softplus

```python
class mindspore.nn.probability.bijector.Softplus(
    sharpness=1.0,
    name="Softplus"
)
```

For more information, see [mindspore.nn.probability.bijector.Softplus](https://www.mindspore.cn/docs/en/r1.7/api_python/nn_probability/mindspore.nn.probability.bijector.Softplus.html).

## Differences

TensorFlow: The formula: $Y = c\*g(X/c) = c\*Log[1 + exp(X/c)] $, hinge_softness = c.

MindSpore：The formula: $Y = g(X) = log(1 + e ^ {kX}) / k $, sharpness = k. Therefore, when sharpness = 1.0/hinge_softness, the calculation results of MindSpore and TensorFlow are equal.

## Code Example

```python
# The following implements bijector.Softplus with MindSpore.
import tensorflow as tf
import tensorflow_probability.python as tfp
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.nn.probability.bijector as msb

# To initialize a Softplus bijector of sharpness 2.0.
softplus = msb.Softplus(2.0)
value = Tensor([2], dtype=mindspore.float32)
ans1 = softplus.forward(value)
print(ans1)
#Out:
#[2.009075]
ans2 = softplus.inverse(value)
print(ans2)
#Out:
#[1.9907573]
ans3 = softplus.forward_log_jacobian(value)
print(ans3)
#Out:
#[-0.01814996]
ans4 = softplus.inverse_log_jacobian(value)
print(ans4)
#Out:
#[0.01848531]


# The following implements bijectors.Softplus with TensorFlow_Probability.
value_tf = tf.constant([2], dtype=tf.float32)
# sharpness = 2.0, sharpness = 1./hinge_softness, so hinge_softness = 0.5
output = tfp.bijectors.Softplus(0.5)
out1 = output.forward(value_tf)
out2 = output.inverse(value_tf)
out3 = output.forward_log_det_jacobian(value_tf, event_ndims=0)
out4 = output.inverse_log_det_jacobian(value_tf, event_ndims=0)
ss = tf.Session()
ss.run(out1)
# out1
# array([2.009075], dtype=float32)
ss.run(out2)
# out2
# array([1.9907573], dtype=float32)
ss.run(out3)
# out3
# array([-0.01814996], dtype=float32)
ss.run(out4)
# out4
# array([0.01848542], dtype=float32)
```
