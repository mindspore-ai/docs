[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/generative/diffusion.md)

# Diffusion Model

This document is based on [Hugging Face: The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) and [Understanding Diffusion Model](https://zhuanlan.zhihu.com/p/525106459).

> This tutorial is successfully executed on Jupyter Notebook. If you download this document as a Python file, ensure that the GUI is installed before executing the Python file.

There are many explanations of diffusion models. This document will introduce it based on denoising diffusion probabilistic model (DDPM). Many remarkable results about DDPM have been achieved for (un)conditional image/audio/video generation. Popular examples include [GLIDE](https://arxiv.org/abs/2112.10741) and [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg and [Image Generation](https://imagen.research.google/) by Google Brain.

Actually, the idea of diffusion-based generative models was already introduced by [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585). [Song et al., 2019](https://arxiv.org/abs/1907.05600) (at Stanford University) and [Ho et al., 2020](https://arxiv.org/abs/2006.11239) (at Google Brain) independently improve the method.

The method stated in this document is achieved on MindSpore AI framework and refers to Phil Wang's [Denoising Diffusion Probabilistic Model, in PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch) (which is achieved based on [TensorFlow](https://github.com/hojonathanho/diffusion)).

![Image-1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/generative/images/diffusion_1.png)

We adopt the discrete time (potential variable model) in the experiment. In addition, you can see [other opinions](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw) on diffusion models.

Before the experiment, install and import the required libraries (assuming you have installed [MindSpore](https://mindspore.cn/install), download, dataset, matplotlib, and tqdm).

```python
import math
from functools import partial
%matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from multiprocessing import cpu_count
from download import download

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.dataset.vision import Resize, Inter, CenterCrop, ToTensor, RandomHorizontalFlip, ToPIL
from mindspore.common.initializer import initializer
from mindspore.amp import DynamicLossScaler

ms.set_seed(0)
```

## Model Introduction

### Diffusion Model

A diffusion model is not complex if you compare it to other generative models such as Normalizing Flows, GANs, or VAEs which convert noise from some simple distribution to a data sample. A diffusion model learns to gradually denoise pure noise through a neural network to generate an actual image.
Processing images using a diffusion model consists of 2 processes.

- A fixed (or predefined) forward diffusion process $q$ that gradually adds Gaussian noise to an image to obtain pure noise.

- A reverse denoising diffusion process $p_\theta$ that learns to gradually denoise pure noise through a neural network to generate an actual image.

![Image-2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/generative/images/diffusion_2.png)

Both the forward and reverse processes indexed by $t$ occur within the number of limited time steps $T$ (the DDPM authors use $T = 1000$). We start with $t=0$, sample the real image $\mathbf{x}_0$ from the data distribution. A cat image from ImageNet is used to show the forward diffusion process, which samples some noise from a Gaussian distribution at each time step $t$ and adds the noise to the image of the previous time step. Assume that a sufficiently large $T$ and a well behaved schedule for adding noise at each time step, you will end up with what is called an [Isotropic Gaussian Distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) at $t = T$ via a gradual process.

### Implementation Principle of the Diffusion Model

#### Forward Diffusion Process

The forward diffusion process is to add Gaussian noise to an image. Although images cannot be generated at this step, it is critical to understand diffusion models and build training samples.
First, we need a controllable loss function and optimize the function by a neural network.

Assume that $q(x_0)$ is a real data distribution. Because of $x_0 \sim q(x_0)$, we can sample from this distribution to obtain the image $x_0$. Next, we define the forward diffusion process $q(x_t | x_{t-1})$. In the forward process, we add Gaussian noise at each time step t based on the known variance ${0}<\beta{1}<\beta{2}<... <\beta_{T}<{1}$. Each step t is related only to the step t-1. Therefore the process may also be considered as a Markov process.

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

The normal distribution (also known as Gaussian distribution) is defined by two parameters: a mean $\mu$ and a variance $\sigma^2 \geq 0$. Basically, each new (slightly noised) image at time step $t$ is drawn from a conditional Gaussian distribution, where:

$$
q(\mathbf{\mu}_t) = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}
$$

We can sample $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and then set:

$$
q(\mathbf{x}_t) = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}
$$

Note that $\beta_t$ is not constant at each time step $t$ (hence the subscript). In fact, we define a so-called "dynamic variance" method, so that $\beta_t$ of each time step can be linear, quadratic, cosine, etc. (a bit like dynamic learning rate method).

Therefore, if we set the schedule properly, starting from $\mathbf{x}_0$, we will end up with $\mathbf{x}_1, ..., \mathbf{x} _t, ..., \mathbf{x}_T$. That is, as $t$ increases, $\mathbf{x}_t$ becomes more similar to pure noise. $\mathbf{x}_T$ is the pure Gaussian noise.

If we know the conditional probability distribution $p(\mathbf{x} _{t-1} | \mathbf{x}_t)$, we can run the process reversely: sample some random Gaussian noise $\mathbf{x}_T$, and then gradually denoise it so that we end up with a sample in the real distribution $\mathbf{x}_0$. However, we do not know the conditional probability distribution $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$. This is intractable since it requires the distribution of all possible images in order to calculate this conditional probability.

#### Reverse Diffusion Process

To solve the preceding problem, a neural network is used to approximate (or learn) the conditional probability distribution $p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)$, where $\theta$ is a parameter of the neural network. If we say the forward diffusion process is a noise adding process, the reverse diffusion process is a denoising process that uses a neural network to represent the reverse process $p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)$.

Now we need a neural network to represent a (conditional) probability distribution of the reverse process. If we assume this reverse process is Gaussian as well, then any Gaussian distribution is defined by 2 parameters:

- A mean parametrized by $\mu_\theta$.

- A variance parametrized by $\mu_\theta$.

We can formulate the process as follows:

$$
p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))
$$

The mean and variance are determined by the noise level $t$. Therefore, the neural network needs to learn and represent the mean and variance.

- However, the DDPM authors decided to keep the variance fixed, and let the neural network only learn (represent) the mean $\mu_\theta$​ of this conditional probability distribution.

- In this document, we also assume that the neural network only needs to learn (represent) the mean $\mu_\theta$ of this conditional probability distribution.

To derive an objective function to learn the mean of the reverse process, the authors observe that the combination of $q$ and $p_\theta$ can be seen as a variational auto-encoder (VAE). Thus, the variational lower bound (also called evidence lower bound, ELBO) can be used to minimize the negative log-likelihood with respect to ground truth data sample $\mathbf{x}_0$ (For more information about ELBO, see the VAE paper [(Kingma et al., 2013)](https://arxiv.org/abs/1312.6114)). It turns out that the ELBO for this process is a sum of losses at each time step $L=L_0+L_1+...+L_T$. Each term (except for $L_0$​) of the loss is actually the KL divergence between 2 Gaussian distributions which can be written explicitly as an L2-loss relative to the means.

As Sohl-Dickstein et al. show, a direct consequence of the constructed forward process is that we can sample $\mathbf{x}_t$ at any arbitrary noise level conditioned on $\mathbf{x}_0$ (because sums of Gaussians are also Gaussian). It is very convenient to sample $\mathbf{x}_t$ without applying $q$ repeatedly. If we use:

$$
\\\alpha_t := 1 - \beta_t\\\\\bar{\alpha}t := \Pi_{s=1}^{t} \alpha_s\\
$$

We will obtain the following result:

$$  
q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})
$$

This means that we can sample the Gaussian noise and scale it properly, and then add it to $\mathbf{x}_0$ to obtain $\mathbf{x}_t$ directly.

Note that $\bar{\alpha}_t$ is a function of the known $\beta_t$ variance schedule and therefore is also known and can be calculated in advance. This allows us to optimize random terms of the loss function $L$ during training. Or in other words, to randomly sample $t$ during training and optimize $L_t$.

As Ho et al. show, another advantage of this property is that the mean can be re-parametrized so that the neural network can learn (predict) the added noise in the KL terms which constitute the losses. This means that our neural network becomes a noise predictor, rather than a (direct) mean predictor. The mean can be calculated as follows:

$$ \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right) $$

The final objective function ${L}_{t}$ is as follows (for a random time step t given by $({\epsilon} \sim N(\mathbf{0}, \mathbf{I}))$).

$$ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}, t) \|^2$$

$\mathbf{x}_0$ is the initial (real and undamaged) image here, $\mathbf{\epsilon}$ is the pure noise sampled at the time step $t$, and $\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)$ is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.

The training algorithm is shown as follows:

![Image-3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/generative/images/diffusion_3.png)

In other words:

- We randomly select a sample $q(\mathbf{x}_0)$ from the real unknown and possibly complex data distribution.

- We sample a noise level $t$ (that is., random time step) uniformly between $1$ and $T$.

- We sample some noise from a Gaussian distribution and destroy the input at the time step $t$ by using the nice property defined above.

- The neural network is trained to predict this noise based on the damaged image $\mathbf{x}_t$, that is, the applied noise is based on the known schedule $\mathbf{x}_t$.

In fact, all of these operations are done by using stochastic gradient descent on batches of data to optimize neural networks.

#### Noise Prediction Using the U-Net Neural Network

The neural network needs to receive a noised image at a specific time step and return the predicted noise. Note that the predicted noise is a tensor with the same size/resolution as the input image. So technically, the network receives and outputs tensors of the same shape. What type of neural network can we use to achieve it?

What we always use here is very similar to the [autoencoder](https://en.wikipedia.org/wiki/Autoencoder). Between the encoder and decoder, the autoencoder has a bottleneck layer. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to retain only the most important information in the bottleneck layer.

As for model architecture, the DDPM authors chose U-Net, which is introduced by [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597) and got the highest level achievement in medical image segmentation at that time. Like any autoencoder, this network consists of a bottleneck in the middle, ensuring that the network learns only the most important information. Importantly, it introduces residual connections between the encoder and decoder, greatly improving gradient flows (which is inspired by [He et al., 2015](https://arxiv.org/abs/1512.03385)).

![Image-4](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/generative/images/diffusion_4.jpg)

We can see that the U-Net model downsamples the input (that is, makes the input smaller in terms of spatial resolution), and then performs upsampling.

## Building a Diffusion Model

Here we will explain each step of building a diffusion model.

First, define some helper functions and classes that will be used when implementing the neural network.

```python
def rearrange(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.reshape((b, head, c, x * y))

def rsqrt(x):
    res = ops.sqrt(x)
    return ops.inv(res)

def randn_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    res = ops.standard_normal(x.shape).astype(dtype)
    return res

def randn(shape, dtype=None):
    if dtype is None:
        dtype = ms.float32
    res = ops.standard_normal(shape).astype(dtype)
    return res

def randint(low, high, size, dtype=ms.int32):
    res = ops.uniform(size, Tensor(low, dtype), Tensor(high, dtype), dtype=dtype)
    return res

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def _check_dtype(d1, d2):
    if ms.float32 in (d1, d2):
        return ms.float32
    if d1 == d2:
        return d1
    raise ValueError('dtype is not supported.')

class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
```

Next, define aliases for upsampling and downsampling operations.

```python
def Upsample(dim):
    return nn.Conv2dTranspose(dim, dim, 4, 2, pad_mode="pad", padding=1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, pad_mode="pad", padding=1)
```

### Position Embeddings

Because the parameters of the neural network are shared across time (noise level), authors apply sinusoidal position embeddings to encode $t$, which inspired by Transformer [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). This makes the neural network "know" at which specific time step (noise level) it is operating, for each image in a batch.

The `SinusoidalPositionEmbeddings` module takes a tensor of the `(batch_size, 1)` shape as input (that is, the noise levels of several noisy images in a batch) and converts it into a tensor with the `(batch_size, dim)` shape, where `dim` is the size of the position embeddings. This is added to each residual block then.

```python
class SinusoidalPositionEmbeddings(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * - emb)
        self.emb = Tensor(emb, ms.float32)

    def construct(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = ops.concat((ops.sin(emb), ops.cos(emb)), axis=-1)
        return emb
```

### ResNet/ConvNeXT Block

Then, define the core building block of the U-Net model. DDPM authors apply a Wide ResNet block ([Zagoruyko et al., 2016](https://arxiv.org/abs/1605.07146)), but Phil Wang decides to replace ResNet with ConvNeXT ([Liu et al., 2022](https://arxiv.org/abs/2201.03545)) because the latter has achieved great success in the image field.

You can choose ResNet or ConvNeXT for the U-Net model. In this document, the ConvNeXT block is selected to build the U-Net model.

```python
class Block(nn.Cell):
    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, pad_mode="pad", padding=1)
        self.proj = c(dim, dim_out, 3, padding=1, pad_mode='pad')
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def construct(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ConvNextBlock(nn.Cell):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.SequentialCell(nn.GELU(), nn.Dense(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, group=dim, pad_mode="pad")
        self.net = nn.SequentialCell(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1, pad_mode="pad"),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1, pad_mode="pad"),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def construct(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            condition = condition.expand_dims(-1).expand_dims(-1)
            h = h + condition

        h = self.net(h)
        return h + self.res_conv(x)
```

### Attention Module

Next, define the Attention module, which is added by DDPM authors between convolutional blocks. Attention is a well-known Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) and has achieved great success in various fields of AI, from NLP to [protein folding](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology). Phil Wang applies two attention variants: one is the conventional multi-head self-attention variant (as used in Transformer), and the other is the [linear attention variant](https://github.com/lucidrains/linear-attention-transformer) ([Shen et al., 2018](https://arxiv.org/abs/1812.01243)), whose time and memory require linear scaling in the sequence length rather than scaling in conventional attention.
For more details about the attention mechanism, please refer to Jay Allamar's [blog](https://jalammar.github.io/illustrated-transformer/).

```python
class Attention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias=True)
        self.map = ops.Map()
        self.partial = ops.Partial()

    def construct(self, x):
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, 1)
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = q * self.scale

        # 'b h d i, b h d j -> b h i j'
        sim = ops.bmm(q.swapaxes(2, 3), k)
        attn = ops.softmax(sim, axis=-1)
        # 'b h i j, b h d j -> b h i d'
        out = ops.bmm(attn, v.swapaxes(2, 3))
        out = out.swapaxes(-1, -2).reshape((b, -1, h, w))

        return self.to_out(out)


class LayerNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.g = Parameter(initializer('ones', (1, dim, 1, 1)), name='g')

    def construct(self, x):
        eps = 1e-5
        var = x.var(1, keepdims=True)
        mean = x.mean(1, keep_dims=True)
        return (x - mean) * rsqrt((var + eps)) * self.g


class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias=False)

        self.to_out = nn.SequentialCell(
            nn.Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias=True),
            LayerNorm(dim)
        )

        self.map = ops.Map()
        self.partial = ops.Partial()

    def construct(self, x):
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, 1)
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = ops.softmax(q, -2)
        k = ops.softmax(k, -1)

        q = q * self.scale
        v = v / (h * w)

        # 'b h d n, b h e n -> b h d e'
        context = ops.bmm(k, v.swapaxes(2, 3))
        # 'b h d e, b h d n -> b h e n'
        out = ops.bmm(context.swapaxes(2, 3), q)

        out = out.reshape((b, -1, h, w))
        return self.to_out(out)
```

### Group Normalization

DDPM authors normalize convolution/attention layers and groups of U-Net ([Wu et al., 2018](https://arxiv.org/abs/1803.08494)). Define a `PreNorm` class that will be used to apply groupnorm before the attention layer.

```python
class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def construct(self, x):
        x = self.norm(x)
        return self.fn(x)
```

### Conditional U-Net

All building blocks (position embeddings, ResNet/ConvNeXT blocks, attention, and group normalization) are defined, it is time to define the entire neural network. The job of the network $\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)$ is to receive a batch of noised images and their noise levels, and output the noise added to the input.

More specifically:
The network obtains a batch of noised images of the `(batch_size, num_channels, height, width)` shape and a batch of noise levels of the `(batch_size, 1)` shape as input, and returns a tensor of the `(batch_size, num_channels, height, width)` shape.

The network building process is as follows:

- First, apply a convolutional layer on the batch of noised images and calculate position embeddings for noise levels.

- Then, apply a sequence of downsampling stages. Each downsampling stage consists of two ResNet/ConvNeXT blocks, groupnorm, attention, residual connection, and a downsampling operation.

- Apply the ResNet or ConvNeXT block at the middle of the network again and interleave it with attention.

- Next, apply a sequence of upsampling stages. Each upsampling stage consists of two ResNet/ConvNeXT blocks, groupnorm, attention, residual connections, and an upsampling operation.

- Finally, apply the ResNet/ConvNeXT blocks and then the convolutional layer.

Eventually, neural networks stack up layers as if they were LEGO blocks (but it is important to [understand how they work](http://karpathy.github.io/2019/04/25/recipe/)).

```python
class Unet(nn.Cell):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            convnext_mult=2,
    ):
        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3, pad_mode="pad", has_bias=True)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ConvNextBlock, mult=convnext_mult)

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.SequentialCell(
                SinusoidalPositionEmbeddings(dim),
                nn.Dense(dim, time_dim),
                nn.GELU(),
                nn.Dense(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.CellList([])
        self.ups = nn.CellList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.CellList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.CellList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.SequentialCell(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def construct(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        len_h = len(h) - 1
        for block1, block2, attn, upsample in self.ups:
            x = ops.concat((x, h[len_h]), 1)
            len_h -= 1
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        return self.final_conv(x)
```

### Forward Diffusion

We already know that the forward diffusion process gradually adds noise to an image from the real distribution in several time steps $T$ and is performed according to a variance schedule. The original DDPM authors adopt a linear schedule as follows:

- Set the forward process variances to constants, linearly increasing from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.

- However, it is shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)) that better results can be obtained when a cosine schedule is used.

Below, we define various schedules of $T$ time steps.

```python
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)
```

First, use a linear schedule with $T=200$ time steps and define variables from $\\β_t$ that we will need, such as the cumulative product of variances $\bar{\alpha}_t$. Each of the following variables is just a one-dimensional tensor that stores values from $t$ to $T$. Importantly, we also define the `extract` function, which will allow us to extract an appropriate batch of $t$ indexes.

```python
# Set the time steps to 200.
timesteps = 200

# Define a beta schedule.
betas = linear_beta_schedule(timesteps=timesteps)

# Define alphas.
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

sqrt_recip_alphas = Tensor(np.sqrt(1. / alphas))
sqrt_alphas_cumprod = Tensor(np.sqrt(alphas_cumprod))
sqrt_one_minus_alphas_cumprod = Tensor(np.sqrt(1. - alphas_cumprod))

# Calculate q(x_{t-1} | x_t, x_0).
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

p2_loss_weight = (1 + alphas_cumprod / (1 - alphas_cumprod)) ** -0.
p2_loss_weight = Tensor(p2_loss_weight)

def extract(a, t, x_shape):
    b = t.shape[0]
    out = Tensor(a).gather(t, -1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
```

We'll use a cat image to illustrate how noise is added at each time step of the diffusion process.

```python
# Download the cat image.
url = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/image_cat.zip'
path = download(url, './', kind="zip", replace=True)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/image_cat.zip (170 kB)

file_sizes: 100%|████████████████████████████| 174k/174k [00:00<00:00, 1.45MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

```python
from PIL import Image

image = Image.open('./image_cat/jpg/000000039769.jpg')
base_width = 160
image = image.resize((base_width, int(float(image.size[1]) * float(base_width / float(image.size[0])))))
image.show()
```

Noise is added to the MindSpore tensors, not the pillow image. First, define the image transformations that allow us to transform from a PIL image to a MindSpore tensor (on which we can add noise), and vice versa.

These transformations are fairly simple: normalize images by dividing them by $255$ (to make them within the range of $[0,1]$), and then make sure they are in the range of $[-1, 1]$. For that, DPPM paper has introduced as follows:

> Assume that image data consists of integers in $\{0, 1, ... , 255\}$ scaled linearly to $[−1, 1]$. This ensures that the neural network reverse process operates on consistently scaled inputs starting from the standard normal prior $p(\mathbf{x}_T )$.

```python
from mindspore.dataset import ImageFolderDataset

image_size = 128
transforms = [
    Resize(image_size, Inter.BILINEAR),
    CenterCrop(image_size),
    ToTensor(),
    lambda t: (t * 2) - 1
]


path = './image_cat'
dataset = ImageFolderDataset(dataset_dir=path, num_parallel_workers=cpu_count(),
                             extensions=['.jpg', '.jpeg', '.png', '.tiff'],
                             num_shards=1, shard_id=0, shuffle=False, decode=True)
dataset = dataset.project('image')
transforms.insert(1, RandomHorizontalFlip())
dataset_1 = dataset.map(transforms, 'image')
dataset_2 = dataset_1.batch(1, drop_remainder=True)
x_start = next(dataset_2.create_tuple_iterator())[0]
print(x_start.shape)
```

```text
(1, 3, 128, 128)
```

Define the reverse transform, which takes in a tensor containing $[-1, 1]$ and transforms them back to the PIL image.

```python
import numpy as np

reverse_transform = [
    lambda t: (t + 1) / 2,
    lambda t: ops.permute(t, (1, 2, 0)), # CHW to HWC
    lambda t: t * 255.,
    lambda t: t.asnumpy().astype(np.uint8),
    ToPIL()
]

def compose(transform, x):
    for d in transform:
        x = d(x)
    return x
```

Let's verify this:

```python
reverse_image = compose(reverse_transform, x_start[0])
reverse_image.show()
```

We can now define the forward diffusion process, as shown in this document:

```python
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = randn_like(x_start)
    return (extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
```

Test it at a specific time step.

```python
def get_noisy_image(x_start, t):
    # Add noise.
    x_noisy = q_sample(x_start, t=t)

    # Transform to a PIL image.
    noisy_image = compose(reverse_transform, x_noisy[0])

    return noisy_image
```

```python
# Sets the time step.
t = Tensor([40])
noisy_image = get_noisy_image(x_start, t)
print(noisy_image)
noisy_image.show()
```

```text
<PIL.Image.Image image mode=RGB size=128x128 at 0x7F54569F3950>
```

Visualize this for different time steps.

```python
import matplotlib.pyplot as plt

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    _, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
```

```python
plot([get_noisy_image(x_start, Tensor([t])) for t in [0, 50, 100, 150, 199]])
```

This means that we can now define the loss function for a given model as follows:

```python
def p_losses(unet_model, x_start, t, noise=None):
    if noise is None:
        noise = randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = unet_model(x_noisy, t)

    loss = nn.SmoothL1Loss()(noise, predicted_noise)# todo
    loss = loss.reshape(loss.shape[0], -1)
    loss = loss * extract(p2_loss_weight, t, loss.shape)
    return loss.mean()
```

The `denoise_model` will be the U-Net defined above. Huber loss will be applied between the real noise and predicted noise.

## Data Preparation and Processing

Here we define a regular dataset. The dataset is composed of images from a simple real dataset, such as Fashion-MNIST, CIFAR-10, or ImageNet, scaled linearly to $[-1, 1]$.

Each image is resized to the same size. Interestingly, the image is also randomly flipped horizontally. According to the paper, we use random horizontal flips during CIFAR10 training and try training both with and without flipping, and find flipping can improve sample quality slightly.

Here the Fashion_MNIST dataset is downloaded and decompressed to a specified path. This dataset consists of images that already have the same resolution, that is, 28 x 28.

```python
# Download the MNIST dataset.
url = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/dataset.zip'
path = download(url, './', kind="zip", replace=True)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/dataset.zip (29.4 MB)

file_sizes: 100%|██████████████████████████| 30.9M/30.9M [00:00<00:00, 43.4MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

```python
from mindspore.dataset import FashionMnistDataset

image_size = 28
channels = 1
batch_size = 16

fashion_mnist_dataset_dir = "./dataset"
dataset = FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir, usage="train", num_parallel_workers=cpu_count(), shuffle=True, num_shards=1, shard_id=0)
```

Next, define a transform operation that will be dynamically applied to the entire dataset. This operation applies some basic image preprocessing: random horizontal flipping, rescaling, and finally making their values in the $[-1,1]$ range.

```python
transforms = [
    RandomHorizontalFlip(),
    ToTensor(),
    lambda t: (t * 2) - 1
]


dataset = dataset.project('image')
dataset = dataset.shuffle(64)
dataset = dataset.map(transforms, 'image')
dataset = dataset.batch(16, drop_remainder=True)
```

```python
x = next(dataset.create_dict_iterator())
print(x.keys())
```

```text
dict_keys(['image'])
```  

### Sampling

Since we will sample from the model during training (to track progress), we define the following code: Sampling is summarized in this document as algorithm 2.

![Image-5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/generative/images/diffusion_5.png)

Generating a new image from a diffusion model is achieved by reversing the diffusion process: starting with $T$, we sample pure noise from the Gaussian distribution, and then use our neural network to gradually denoise (using the conditional probability it learns), until we finally end up at the time step $t = 0$. As shown above, we can derive a slightly less denoised image $\mathbf{x}_{t-1 }$ by plugging in the reparametrization of the mean,
using our noise predictor. Note that the variance is known in advance.

Ideally, we will end up with an image that looks like it comes from a real data distribution.

The following code implements this.

```python
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    noise = randn_like(x)
    return model_mean + ops.sqrt(posterior_variance_t) * noise

def p_sample_loop(model, shape):
    b = shape[0]
    # Start with the pure noise.
    img = randn(shape, dtype=None)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, ms.numpy.full((b,), i, dtype=mstype.int32), i)
        imgs.append(img.asnumpy())
    return imgs

def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
```

Note that the above code is a simplified version of the original implementation.

## Training Process

Now, let's start training.

```python
# Defining a dynamic learning rate.
lr = nn.cosine_decay_lr(min_lr=1e-7, max_lr=1e-4, total_step=10*3750, step_per_epoch=3750, decay_epoch=10)

# Defining a U-Net model.
unet_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)

name_list = []
for (name, par) in list(unet_model.parameters_and_names()):
    name_list.append(name)
i = 0
for item in list(unet_model.trainable_params()):
    item.name = name_list[i]
    i += 1

# Define an optimizer.
optimizer = nn.Adam(unet_model.trainable_params(), learning_rate=lr)
loss_scaler = DynamicLossScaler(65536, 2, 1000)

# Define the forward process.
def forward_fn(data, t, noise=None):
    loss = p_losses(unet_model, data, t, noise)
    return loss

# Calculate the gradient.
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

# Update the gradient.
def train_step(data, t, noise):
    loss, grads = grad_fn(data, t, noise)
    optimizer(grads)
    return loss
```

```python
import time

epochs = 10

iterator = dataset.create_tuple_iterator(num_epochs=epochs)
for epoch in range(epochs):
    begin_time = time.time()
    for step, batch in enumerate(iterator):
        unet_model.set_train()
        batch_size = batch[0].shape[0]
        t = randint(0, timesteps, (batch_size,), dtype=ms.int32)
        noise = randn_like(batch[0])
        loss = train_step(batch[0], t, noise)

        if step % 500 == 0:
            print(" epoch: ", epoch, " step: ", step, " Loss: ", loss)
    end_time = time.time()
    times = end_time - begin_time
    print("training time:", times, "s")
    # Display the random sampling effect.
    unet_model.set_train(False)
    samples = sample(unet_model, image_size=image_size, batch_size=64, channels=channels)
    plt.imshow(samples[-1][5].reshape(image_size, image_size, channels), cmap="gray")
print("Training Success!")
```

```text
 epoch:  0  step:  0  Loss:  0.43375123
 epoch:  0  step:  500  Loss:  0.113769315
 epoch:  0  step:  1000  Loss:  0.08649178
 epoch:  0  step:  1500  Loss:  0.067664884
 epoch:  0  step:  2000  Loss:  0.07234038
 epoch:  0  step:  2500  Loss:  0.043936778
 epoch:  0  step:  3000  Loss:  0.058127824
 epoch:  0  step:  3500  Loss:  0.049789283
training time: 922.3438229560852 s
 epoch:  1  step:  0  Loss:  0.05088563
 epoch:  1  step:  500  Loss:  0.051174678
 epoch:  1  step:  1000  Loss:  0.04455947
 epoch:  1  step:  1500  Loss:  0.055165425
 epoch:  1  step:  2000  Loss:  0.043942295
 epoch:  1  step:  2500  Loss:  0.03274461
 epoch:  1  step:  3000  Loss:  0.048117325
 epoch:  1  step:  3500  Loss:  0.063063145
training time: 937.5596783161163 s
 epoch:  2  step:  0  Loss:  0.052893892
 epoch:  2  step:  500  Loss:  0.05721748
 epoch:  2  step:  1000  Loss:  0.057248186
 epoch:  2  step:  1500  Loss:  0.048806388
 epoch:  2  step:  2000  Loss:  0.05007638
 epoch:  2  step:  2500  Loss:  0.04337231
 epoch:  2  step:  3000  Loss:  0.043207955
 epoch:  2  step:  3500  Loss:  0.034530163
training time: 947.6374666690826 s
 epoch:  3  step:  0  Loss:  0.04867614
 epoch:  3  step:  500  Loss:  0.051636297
 epoch:  3  step:  1000  Loss:  0.03338969
 epoch:  3  step:  1500  Loss:  0.0420174
 epoch:  3  step:  2000  Loss:  0.052145053
 epoch:  3  step:  2500  Loss:  0.03905913
 epoch:  3  step:  3000  Loss:  0.07621498
 epoch:  3  step:  3500  Loss:  0.06484105
training time: 957.7780408859253 s
 epoch:  4  step:  0  Loss:  0.046281893
 epoch:  4  step:  500  Loss:  0.03783619
 epoch:  4  step:  1000  Loss:  0.0587488
 epoch:  4  step:  1500  Loss:  0.06974746
 epoch:  4  step:  2000  Loss:  0.04299112
 epoch:  4  step:  2500  Loss:  0.027945498
 epoch:  4  step:  3000  Loss:  0.045338146
 epoch:  4  step:  3500  Loss:  0.06362417
training time: 955.6116819381714 s
 epoch:  5  step:  0  Loss:  0.04781142
 epoch:  5  step:  500  Loss:  0.032488734
 epoch:  5  step:  1000  Loss:  0.061507083
 epoch:  5  step:  1500  Loss:  0.039130375
 epoch:  5  step:  2000  Loss:  0.034972396
 epoch:  5  step:  2500  Loss:  0.039485026
 epoch:  5  step:  3000  Loss:  0.06690869
 epoch:  5  step:  3500  Loss:  0.05355365
training time: 951.7758958339691 s
 epoch:  6  step:  0  Loss:  0.04807706
 epoch:  6  step:  500  Loss:  0.021469856
 epoch:  6  step:  1000  Loss:  0.035354104
 epoch:  6  step:  1500  Loss:  0.044303045
 epoch:  6  step:  2000  Loss:  0.040063944
 epoch:  6  step:  2500  Loss:  0.02970439
 epoch:  6  step:  3000  Loss:  0.041152682
 epoch:  6  step:  3500  Loss:  0.02062454
training time: 955.2220208644867 s
 epoch:  7  step:  0  Loss:  0.029668871
 epoch:  7  step:  500  Loss:  0.028485576
 epoch:  7  step:  1000  Loss:  0.029675964
 epoch:  7  step:  1500  Loss:  0.052743085
 epoch:  7  step:  2000  Loss:  0.03664278
 epoch:  7  step:  2500  Loss:  0.04454907
 epoch:  7  step:  3000  Loss:  0.043067697
 epoch:  7  step:  3500  Loss:  0.0619511
training time: 952.6654670238495 s
 epoch:  8  step:  0  Loss:  0.055328347
 epoch:  8  step:  500  Loss:  0.035807922
 epoch:  8  step:  1000  Loss:  0.026412832
 epoch:  8  step:  1500  Loss:  0.051044375
 epoch:  8  step:  2000  Loss:  0.05474911
 epoch:  8  step:  2500  Loss:  0.044595096
 epoch:  8  step:  3000  Loss:  0.034082986
 epoch:  8  step:  3500  Loss:  0.02653109
training time: 961.9374921321869 s
 epoch:  9  step:  0  Loss:  0.039675284
 epoch:  9  step:  500  Loss:  0.046295933
 epoch:  9  step:  1000  Loss:  0.031403508
 epoch:  9  step:  1500  Loss:  0.028816734
 epoch:  9  step:  2000  Loss:  0.06530296
 epoch:  9  step:  2500  Loss:  0.051451046
 epoch:  9  step:  3000  Loss:  0.037913296
 epoch:  9  step:  3500  Loss:  0.030541396
training time: 974.643147945404 s
Training Success!
```

## Inference Process (Sampling from a Model)

To sample from a model, we can use only the sampling function defined above.

```python
# Sample 64 images.
unet_model.set_train(False)
samples = sample(unet_model, image_size=image_size, batch_size=64, channels=channels)
```

```python
# Display a random one.
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
```

```text
<matplotlib.image.AxesImage at 0x7f5175ea1690>
```

You can see that this model can generate a piece of clothing!

Note that the resolution of the dataset we train is quite low (28 x 28).

You can also create a GIF file for the denoising process.

```python
import matplotlib.animation as animation

random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
animate.save('diffusion.gif')
plt.show()
```

## Conclusion

Note that DDPM papers show that diffusion models are a promising direction for (un)conditional image generation. Since then, diffusion has been (greatly) improved, most notably for text-conditional image generation. The following are some important follow-up works:

- Improved Denoising Diffusion Probabilistic Models ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)): finds that learning the variance of the conditional distribution (except the mean) facilitates the performance.

- Cascaded Diffusion Models for High Fidelity Image Generation ([Ho et al., 2021](https://arxiv.org/abs/2106.15282)): introduces cascaded diffusion, which includes a pipeline of multiple diffusion models. These models generate images with improved resolution for high-fidelity image synthesis.

- Diffusion Models Beat GANs on Image Synthesis ([Dhariwal et al., 2021](https://arxiv.org/abs/2105.05233)): shows that the diffusion model can obtain better image sample quality than that generated by the most advanced model by improving the U-Net architecture and introducing classifier guidance.

- Classifier-Free Diffusion Guidance ([Ho et al., 2021](https://openreview.net/pdf?id=qw8AKxfYbI)): shows that no classifier is required to guide a diffusion model by jointly training a conditional and an unconditional diffusion model with a single neural network.

- Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2) ([Ramesh et al., 2022](https://cdn.openai.com/papers/dall-e-2.pdf)): uses a prior to turn a text caption into a CLIP image embedding, after which a diffusion model decodes it into an image.

- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (ImageGen) ([Saharia et al., 2022](https://arxiv.org/abs/2205.11487)): shows that combining large pre-trained language models (such as T5) with cascade diffusion are effective for text-to-image synthesis.

Note that this list includes only important works prior to the writing of this document, which is. June 7, 2022.

Currently, the main (perhaps the only) drawback of diffusion models is that they require multiple forward passes to generate images (which is not the case for generative models, such as GAN). However, [ongoing research](https://arxiv.org/abs/2204.13902) shows that only 10 denoising steps are required to achieve high-fidelity generation.

## References

1. [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

2. [Understanding Diffusion Model](https://zhuanlan.zhihu.com/p/525106459)
