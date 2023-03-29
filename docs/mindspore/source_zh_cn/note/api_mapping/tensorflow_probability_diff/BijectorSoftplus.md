# 比较与tfp.bijectors.Softplus的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_probability_diff/BijectorSoftplus.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## tfp.bijectors.Softplus

```python
class tfp.bijectors.Softplus(
    hinge_softness=None,
    low=None,
    validate_args=False,
    name='softplus'
)
```

更多内容详见[tfp.bijectors.Softplus](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Softplus)。

## mindspore.nn.probability.bijector.Softplus

```python
class mindspore.nn.probability.bijector.Softplus(
    sharpness=1.0,
    name="Softplus"
)
```

更多内容详见[mindspore.nn.probability.bijector.Softplus](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn_probability/mindspore.nn.probability.bijector.Softplus.html)。

## 使用方式

TensorFlow：公式：$Y = c\*g(X/c) = c\*Log[1 + exp(X/c)] $，hinge_softness = c。

MindSpore：公式：$Y = g(X) = log(1 + e ^ {kX}) / k $，sharpness = k。所以当sharpness = 1.0/hinge_softness的时候，MindSpore与TensorFlow的计算结果是一致的。

## 代码示例

```python
# The following implements bijector.Softplus with MindSpore.
import tensorflow as tf
import tensorflow_probability.python as tfp
import mindspore as ms
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb

# To initialize a Softplus bijector of sharpness 2.0.
softplus = msb.Softplus(2.0)
value = ms.Tensor([2], dtype=ms.float32)
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
