# Per-sample-gradients

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_en/func_programming/per_sample_gradients.md)

Calculating per-sample-gradients means calculating the gradient of each sample in a batch sample. When training a neural network, many deep learning frameworks calculate the gradients of the batch samples and use the gradients of the batch samples to update the network parameters. per-sample-gradients can help us to better improve the training of the model by more accurately calculating the effect of each sample on the network parameters when training the neural network.

Calculating per-sample-gradients is a troblesome business in many deep learning computational frameworks because these frameworks directly accumulate the gradients of the entire batch of samples. Using these frameworks, we can think of a simple way to compute per-sample-gradients, i.e., to compute the loss of the predicted and labeled values for each of the batch samples and to compute the gradient of that loss with respect to the network parameters, but this method is clearly very inefficient.

MindSpore provides us with a more efficient way to calculate per-sample-gradients.

We illustrate the efficient method of computing per-sample-gradients with the example of TD(0) (Temporal Difference) algorithm, which is a reinforcement learning algorithm based on temporal difference that learns the optimal strategy in the absence of an environment model. In the TD(0) algorithm, the valued function estimates are updated according to the current rewards. The TD(0) algorithm is formulated as follows:

$$V(S_{t}) = V(S_{t}) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_{t}))$$

where $V(S_{t})$ is the current valued function estimate, $\alpha$ is the learning rate, $R_{t+1}$ is the reward obtained after performing the action in state $S_{t}$, $\gamma$ is the discount factor, $V(S_{t+1})$ is the valued function estimate for the next state $S_{t+1}$, $R_{t+1} + \ gamma V(S_{t+1})$ is known as the TD target, and $R_{t+1} + \gamma V(S_{t+1}) - V(S_{t})$ is known as the TD bias.

By continuously updating the valued function estimates using the TD(0) algorithm, the optimal policy can be learned incrementally to maximize the reward gained in the environment.

Combining jit, vmap and grad in MindSpore, we get a more efficient way to compute per-sample-gradients.

The method is described below, assuming that the estimate $v_{\theta}$ at state $s_{t}$ is parameterized by a linear function.

```python
from mindspore import ops, Tensor, vmap, jit, grad


value_fn = lambda theta, state: ops.tensor_dot(theta, state, axes=1)
theta = Tensor([0.2, -0.2, 0.1])
```

Consider the following scenario, transforming from state $s_{t}$ to state $s_{t+1}$ and in which we observe a reward of $r_{t+1}$.

```python
s_t = Tensor([2., 1., -2.])
r_tp1 = Tensor(2.)
s_tp1 = Tensor([1., 2., 0.])
```

The updating volume of the parameter ${\theta}$ is given by:

$$\Delta{\theta}=(r_{t+1} + v_{\theta}(s_{t+1}) - v_{\theta}(s_{t}))\nabla v_{\theta}(s_{t})$$

The update of the parameter ${\theta}$ is not the gradient of any loss function, however, it can be considered as the gradient of the following pseudo-loss function (assuming that the effect of the target value $r_{t+1} + v_{\theta}(s_{t+1})$ on the computation of the gradient of $L(\theta)$ with respect to ${\theta}$ is ignored).

$$L(\theta) = [r_{t+1} + v_{\theta}(s_{t+1}) - v_{\theta}(s_{t})]^{2}$$

When computing the update of the parameter ${\theta}$ (computing the gradient of $L(\theta)$ with respect to ${\theta}$), we need to eliminate the effect of the target value $r_{t+1} + v_{\theta}(s_{t+1})$ on the computation of the gradient of ${\theta}$ using `ops.stop_gradient`, which can be made such that the target values $r_{t+1} + v_{\theta}(s_{t+1})$ do not contribute to the derivation of ${\theta}$ during the derivation process to obtain the correct update of the parameter ${\theta}$.

We give the implementation of the pseudo-loss function $L(\theta)$ in MindSpore.

```python
def td_loss(theta, s_tm1, r_t, s_t):
    v_t = value_fn(theta, s_t)
    target = r_tp1 + value_fn(theta, s_tp1)
    return (ops.stop_gradient(target) - v_t) ** 2
```

Pass `td_loss` into `grad` and compute the gradient of td_loss with respect to `theta`, i.e., the update of `theta`.

```python
td_update = grad(td_loss)
delta_theta = td_update(theta, s_t, r_tp1, s_tp1)
print(delta_theta)
```

```text
[-4. -8. -0.]
```

`td_update` computes the gradient of td_loss with respect to the parameter ${\theta}$ based on only one sample. We can vectorize this function using `vmap` which will add a batch dimension to all inputs and outputs. Now, we give a batch of inputs and produce a batch of outputs, with each output element in the output batch corresponding to the corresponding input element in the input batch.

```python
batched_s_t = ops.stack([s_t, s_t])
batched_r_tp1 = ops.stack([r_tp1, r_tp1])
batched_s_tp1 = ops.stack([s_tp1, s_tp1])
batched_theta = ops.stack([theta, theta])

per_sample_grads = vmap(td_update)
batch_theta = ops.stack([theta, theta])
delta_theta = per_sample_grads(batched_theta, batched_s_t, batched_r_tp1, batched_s_tp1)
print(delta_theta)
```

```text
[[-4. -8.  0.]
 [-4. -8.  0.]]
```

In the above example, we need to manually pass a batch of `theta` for `per_sample_grads`, but in reality, we can pass just a single `theta`. To complete this, we pass the parameter `in_axes` to `vmap`, where the position corresponding to the parameter `theta` in `in_axes` is set to `None` and the positions corresponding to the other parameters are set to `0`. This allows us to add an additional axis only to parameters other than `theta`.

```python
inefficiecient_per_sample_grads = vmap(td_update, in_axes=(None, 0, 0, 0))
delta_theta = inefficiecient_per_sample_grads(theta, batched_s_t, batched_r_tp1, batched_s_tp1)
print(delta_theta)
```

```text
[[-4. -8.  0.]
 [-4. -8.  0.]]
```

Up to this point, the gradient for each sample is calculated correctly, but we can also make the calculation process a bit faster. We call `inefficiecient_per_sample_grads` using `jit`, which will compile `inefficiecient_per_sample_grads` into a callable MindSpore graph and improve the efficiency of its operation.

```python
efficiecient_per_sample_grads = jit(inefficiecient_per_sample_grads)
delta_theta = efficiecient_per_sample_grads(theta, batched_s_t, batched_r_tp1, batched_s_tp1)
print(delta_theta)
```

```text
[[-4. -8.  0.]
 [-4. -8.  0.]]
```