[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/generative/cyclegan.md)

# CycleGAN for Image Style Migration

> Running this case requires a large amount of memory. You are advised to run this case on Ascend or GPU.

## Model Introduction

### Overview

Cycle generative adversarial network (CycleGAN) comes from [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593). The model implements a method of learning to translate an image from a source domain X to a target domain Y in the absence of paired examples.

An important application field of this model is domain adaptation, which can be generally understood as image style migration. Before CycleGAN, domain adaptation models, such as Pix2Pix, are available. However, Pix2Pix requires that training data be in pairs. In real life, it is difficult to find images that appear in pairs in two domains (image styles). CycleGAN requires only data in two domains and does not require strict correspondence between them, which is a new unsupervised image migration network.

### Model Structure

Essentially, a CycleGAN consists of two mirror-symmetric GANs. The following figure shows the CycleGAN structure. (The figure comes from the original paper.)

![CycleGAN](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_zh_cn/generative/images/CycleGAN.png)

For ease of understanding, apples and oranges are used as examples. In the preceding figure, $X$ indicates apples, $Y$ indicates oranges, $G$ indicates an apple-to-orange style generator, $F$ indicates an orange-to-apple style generator, and $D_{X}$ and $D_{Y}$ are corresponding discriminators. For details about the structures of the generators and discriminators, see the following code. The model can finally output weights of the two models, and separately migrate styles of the two images to each other to generate new images.

An important part of this model is loss functions, in which the cycle consistency loss is the most important function. The following figure shows the process of calculating the cycle loss. (The figure comes from the original paper.)

![Cycle Consistency Loss](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_zh_cn/generative/images/CycleGAN_1.png)

In the preceding figure, the apple image $x$ passes through the generator $G$ to obtain the pseudo orange $\hat{Y}$, and then sends the pseudo orange $\hat{Y}$ result to the generator $F$ to generate the apple-style result $\hat{x}$. Finally, the generated apple-style result $\hat{x}$ and the original apple image $x$ are used to calculate the cycle consistency loss, and vice versa. Cycle loss captures the intuition that if we translate from one domain to the other and back again we should arrive at where we started. For details about the training process, see the following code.

## Dataset

The images in dataset used in this case come from [ImageNet](https://ieeexplore.ieee.org/document/5206848). The dataset has 17 data packages. This document uses only the apple and orange package. Images are scaled to 256 x 256 pixels, including 996 apple images and 1020 orange images for training and 266 apple images and 248 orange images for testing.

Here, random cropping, horizontal random flipping, and normalization preprocessing are performed on the data. To focus on the model, the data preprocessing result is converted into data in MindRecord format to omit most data preprocessing code.

### Downloading a Dataset

Use the `download` API to download the dataset and decompress it to the current directory. Before downloading data, use `pip install download` to install the `download` package.

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/application/CycleGAN_apple2orange.zip"

download(url, ".", kind="zip", replace=True)
```

### Loading a Dataset

Use the `MindDataset` API of MindSpore to read and parse the dataset.

```python
from mindspore.dataset import MindDataset

# Read data in MindRecord format.
name_mr = "./CycleGAN_apple2orange/apple2orange_train.mindrecord"
data = MindDataset(dataset_files=name_mr)
print("Datasize: ", data.get_dataset_size())

batch_size = 1
dataset = data.batch(batch_size)
datasize = dataset.get_dataset_size()
```

```text
Datasize:  1019
```

### Visualization

Use the `create_dict_iterator` function to convert data into a dictionary iterator, and then use the `matplotlib` module to visualize some training data.

```python
import numpy as np
import matplotlib.pyplot as plt

mean = 0.5 * 255
std = 0.5 * 255

plt.figure(figsize=(12, 5), dpi=60)
for i, data in enumerate(dataset.create_dict_iterator()):
    if i < 5:
        show_images_a = data["image_A"].asnumpy()
        show_images_b = data["image_B"].asnumpy()

        plt.subplot(2, 5, i+1)
        show_images_a = (show_images_a[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))
        plt.imshow(show_images_a)
        plt.axis("off")

        plt.subplot(2, 5, i+6)
        show_images_b = (show_images_b[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))
        plt.imshow(show_images_b)
        plt.axis("off")
    else:
        break
plt.show()
```

## Building Generators

The model structure of generators in this case is the same as that of the ResNet model. According to the original paper, we use 6 residual blocks for 128 x 128 input images and 9 blocks for 256×256 and higher-resolution training images. In this document, 9 residual blocks are connected, and the hyperparameter `n_layers` controls the number of residual blocks.

The structure of the generators is as follows:

![CycleGAN Generator](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_zh_cn/generative/images/CycleGAN_2.jpg)

For details about the model structure, see the following code:

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

weight_init = Normal(sigma=0.02)

class ConvNormReLU(nn.Cell):
    def __init__(self, input_channel, out_planes, kernel_size=4, stride=2, alpha=0.2, norm_mode='instance',
                 pad_mode='CONSTANT', use_relu=True, padding=None, transpose=False):
        super(ConvNormReLU, self).__init__()
        norm = nn.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            norm = nn.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            if transpose:
                conv = nn.Conv2dTranspose(input_channel, out_planes, kernel_size, stride, pad_mode='same',
                                          has_bias=has_bias, weight_init=weight_init)
            else:
                conv = nn.Conv2d(input_channel, out_planes, kernel_size, stride, pad_mode='pad',
                                 has_bias=has_bias, padding=padding, weight_init=weight_init)
            layers = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            if transpose:
                conv = nn.Conv2dTranspose(input_channel, out_planes, kernel_size, stride, pad_mode='pad',
                                          has_bias=has_bias, weight_init=weight_init)
            else:
                conv = nn.Conv2d(input_channel, out_planes, kernel_size, stride, pad_mode='pad',
                                 has_bias=has_bias, weight_init=weight_init)
            layers = [pad, conv, norm]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class ResidualBlock(nn.Cell):
    def __init__(self, dim, norm_mode='instance', dropout=False, pad_mode="CONSTANT"):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode)
        self.conv2 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode, use_relu=False)
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

    def construct(self, x):
        out = self.conv1(x)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        return x + out


class ResNetGenerator(nn.Cell):
    def __init__(self, input_channel=3, output_channel=64, n_layers=9, alpha=0.2, norm_mode='instance', dropout=False,
                 pad_mode="CONSTANT"):
        super(ResNetGenerator, self).__init__()
        self.conv_in = ConvNormReLU(input_channel, output_channel, 7, 1, alpha, norm_mode, pad_mode=pad_mode)
        self.down_1 = ConvNormReLU(output_channel, output_channel * 2, 3, 2, alpha, norm_mode)
        self.down_2 = ConvNormReLU(output_channel * 2, output_channel * 4, 3, 2, alpha, norm_mode)
        layers = [ResidualBlock(output_channel * 4, norm_mode, dropout=dropout, pad_mode=pad_mode)] * n_layers
        self.residuals = nn.SequentialCell(layers)
        self.up_2 = ConvNormReLU(output_channel * 4, output_channel * 2, 3, 2, alpha, norm_mode, transpose=True)
        self.up_1 = ConvNormReLU(output_channel * 2, output_channel, 3, 2, alpha, norm_mode, transpose=True)
        if pad_mode == "CONSTANT":
            self.conv_out = nn.Conv2d(output_channel, 3, kernel_size=7, stride=1, pad_mode='pad',
                                      padding=3, weight_init=weight_init)
        else:
            pad = nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=pad_mode)
            conv = nn.Conv2d(output_channel, 3, kernel_size=7, stride=1, pad_mode='pad', weight_init=weight_init)
            self.conv_out = nn.SequentialCell([pad, conv])

    def construct(self, x):
        x = self.conv_in(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.residuals(x)
        x = self.up_2(x)
        x = self.up_1(x)
        output = self.conv_out(x)
        return ops.tanh(output)

# Instantiate the generator.
net_rg_a = ResNetGenerator()
net_rg_a.update_parameters_name('net_rg_a.')

net_rg_b = ResNetGenerator()
net_rg_b.update_parameters_name('net_rg_b.')

```

## Building Discriminators

A discriminator is actually a binary network model, and outputs a probability of determining that the image is a real image. The network model uses the PatchGANs model whose patch size is 70 x 70. It is processed through a series of `Conv2d`, `BatchNorm2d`, and `LeakyReLU` layers and obtains the final probability through the Sigmoid activation function.

```python
# Define a discriminator.
class Discriminator(nn.Cell):
    def __init__(self, input_channel=3, output_channel=64, n_layers=3, alpha=0.2, norm_mode='instance'):
        super(Discriminator, self).__init__()
        kernel_size = 4
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size, 2, pad_mode='pad', padding=1, weight_init=weight_init),
                  nn.LeakyReLU(alpha)]
        nf_mult = output_channel
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8) * output_channel
            layers.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 2, alpha, norm_mode, padding=1))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) * output_channel
        layers.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 1, alpha, norm_mode, padding=1))
        layers.append(nn.Conv2d(nf_mult, 1, kernel_size, 1, pad_mode='pad', padding=1, weight_init=weight_init))
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output

# Initialize the discriminator.
net_d_a = Discriminator()
net_d_a.update_parameters_name('net_d_a.')

net_d_b = Discriminator()
net_d_b.update_parameters_name('net_d_b.')

```

## Optimizer and Loss Function

The optimizer needs to be set separately based on different models, which is determined by the training process.

For the generator $G$ and its discriminator $D_{Y}$, the target loss function is defined as:

$$L_{GAN}(G,D_Y,X,Y)=E_{y-p_{data}(y)}[logD_Y(y)]+E_{x-p_{data}(x)}[log(1-D_Y(G(x)))]$$

$G$ attempts to generate an image $G(x)$ that looks similar to the image in $Y$, while $D_{Y}$ aims to distinguish the translated sample $G(x)$ from the real sample $y$. The goal of the generator is to minimize this loss function against the discriminator. That is, $ min_{G} max_{D_{Y}}L_{GAN}(G,D_{Y} ,X,Y )$.

A separate adversarial loss cannot ensure that the learned function can map a single input to the expected output. To further reduce the space of the possible mapping function, the learned mapping function should be cycle-consistent. For example, for each image $x$ of $X$, the image translation cycle should be able to bring $x$ back to the original image, which may be referred to as forward cycle consistency. That is, $x→G(x)→F(G(x))\approx x$. For $Y$, it is similar to $x→G(x)→F(G(x))\approx x$. It can be understood that a cycle consistency loss is used to motivate this behavior.

The cycle consistency loss function is defined as follows:

$$L_{cyc}(G,F)=E_{x-p_{data}(x)}[\Vert F(G(x))-x\Vert_{1}]+E_{y-p_{data}(y)}[\Vert G(F(y))-y\Vert_{1}]$$

The cycle consistency loss ensures that the rebuilt image $F(G(x))$ closely matches the input image $x$.

```python
# Build a generator, discriminator, and optimizer.
optimizer_rg_a = nn.Adam(net_rg_a.trainable_params(), learning_rate=0.0002, beta1=0.5)
optimizer_rg_b = nn.Adam(net_rg_b.trainable_params(), learning_rate=0.0002, beta1=0.5)

optimizer_d_a = nn.Adam(net_d_a.trainable_params(), learning_rate=0.0002, beta1=0.5)
optimizer_d_b = nn.Adam(net_d_b.trainable_params(), learning_rate=0.0002, beta1=0.5)

# GAN loss function. The sigmoid function is not used at the last layer.
loss_fn = nn.MSELoss(reduction='mean')
l1_loss = nn.L1Loss("mean")

def gan_loss(predict, target):
    target = ops.ones_like(predict) * target
    loss = loss_fn(predict, target)
    return loss

```

## Forward Computation

Set up a model to compute the loss forward. The process is as follows:

In order to reduce model oscillations [1], the strategy of Shrivastava et al. [2] is followed here to update the discriminator using a history of generated images rather than the ones produced by the latest generator. Here, the `image_pool` function is created, and an image buffer is reserved for storing the 50 images generated by the generator.

```python
import mindspore as ms

# Forward computation

def generator(img_a, img_b):
    fake_a = net_rg_b(img_b)
    fake_b = net_rg_a(img_a)
    rec_a = net_rg_b(fake_b)
    rec_b = net_rg_a(fake_a)
    identity_a = net_rg_b(img_a)
    identity_b = net_rg_a(img_b)
    return fake_a, fake_b, rec_a, rec_b, identity_a, identity_b

lambda_a = 10.0
lambda_b = 10.0
lambda_idt = 0.5

def generator_forward(img_a, img_b):
    true = Tensor(True, dtype.bool_)
    fake_a, fake_b, rec_a, rec_b, identity_a, identity_b = generator(img_a, img_b)
    loss_g_a = gan_loss(net_d_b(fake_b), true)
    loss_g_b = gan_loss(net_d_a(fake_a), true)
    loss_c_a = l1_loss(rec_a, img_a) * lambda_a
    loss_c_b = l1_loss(rec_b, img_b) * lambda_b
    loss_idt_a = l1_loss(identity_a, img_a) * lambda_a * lambda_idt
    loss_idt_b = l1_loss(identity_b, img_b) * lambda_b * lambda_idt
    loss_g = loss_g_a + loss_g_b + loss_c_a + loss_c_b + loss_idt_a + loss_idt_b
    return fake_a, fake_b, loss_g, loss_g_a, loss_g_b, loss_c_a, loss_c_b, loss_idt_a, loss_idt_b

def generator_forward_grad(img_a, img_b):
    _, _, loss_g, _, _, _, _, _, _ = generator_forward(img_a, img_b)
    return loss_g

def discriminator_forward(img_a, img_b, fake_a, fake_b):
    false = Tensor(False, dtype.bool_)
    true = Tensor(True, dtype.bool_)
    d_fake_a = net_d_a(fake_a)
    d_img_a = net_d_a(img_a)
    d_fake_b = net_d_b(fake_b)
    d_img_b = net_d_b(img_b)
    loss_d_a = gan_loss(d_fake_a, false) + gan_loss(d_img_a, true)
    loss_d_b = gan_loss(d_fake_b, false) + gan_loss(d_img_b, true)
    loss_d = (loss_d_a + loss_d_b) * 0.5
    return loss_d

def discriminator_forward_a(img_a, fake_a):
    false = Tensor(False, dtype.bool_)
    true = Tensor(True, dtype.bool_)
    d_fake_a = net_d_a(fake_a)
    d_img_a = net_d_a(img_a)
    loss_d_a = gan_loss(d_fake_a, false) + gan_loss(d_img_a, true)
    return loss_d_a

def discriminator_forward_b(img_b, fake_b):
    false = Tensor(False, dtype.bool_)
    true = Tensor(True, dtype.bool_)
    d_fake_b = net_d_b(fake_b)
    d_img_b = net_d_b(img_b)
    loss_d_b = gan_loss(d_fake_b, false) + gan_loss(d_img_b, true)
    return loss_d_b

# An image buffer is reserved to store the 50 images created previously.
pool_size = 50
def image_pool(images):
    num_imgs = 0
    image1 = []
    if isinstance(images, Tensor):
        images = images.asnumpy()
    return_images = []
    for image in images:
        if num_imgs < pool_size:
            num_imgs = num_imgs + 1
            image1.append(image)
            return_images.append(image)
        else:
            if random.uniform(0, 1) > 0.5:
                random_id = random.randint(0, pool_size - 1)

                tmp = image1[random_id].copy()
                image1[random_id] = image
                return_images.append(tmp)

            else:
                return_images.append(image)
    output = Tensor(return_images, ms.float32)
    if output.ndim != 4:
        raise ValueError("img should be 4d, but get shape {}".format(output.shape))
    return output

```

## Gradient Calculation and Backward Propagation

Gradient calculation is performed based on different models. For details, see the following code.

```python
from mindspore import value_and_grad

# Instantiate the gradient calculation method.
grad_g_a = value_and_grad(generator_forward_grad, None, net_rg_a.trainable_params())
grad_g_b = value_and_grad(generator_forward_grad, None, net_rg_b.trainable_params())

grad_d_a = value_and_grad(discriminator_forward_a, None, net_d_a.trainable_params())
grad_d_b = value_and_grad(discriminator_forward_b, None, net_d_b.trainable_params())

# Calculate the gradient of the generator and backpropagate the update parameters.
def train_step_g(img_a, img_b):
    net_d_a.set_grad(False)
    net_d_b.set_grad(False)

    fake_a, fake_b, lg, lga, lgb, lca, lcb, lia, lib = generator_forward(img_a, img_b)

    _, grads_g_a = grad_g_a(img_a, img_b)
    _, grads_g_b = grad_g_b(img_a, img_b)
    optimizer_rg_a(grads_g_a)
    optimizer_rg_b(grads_g_b)

    return fake_a, fake_b, lg, lga, lgb, lca, lcb, lia, lib

# Calculate the gradient of the discriminator and backpropagate the update parameters.
def train_step_d(img_a, img_b, fake_a, fake_b):
    net_d_a.set_grad(True)
    net_d_b.set_grad(True)

    loss_d_a, grads_d_a = grad_d_a(img_a, fake_a)
    loss_d_b, grads_d_b = grad_d_b(img_b, fake_b)

    loss_d = (loss_d_a + loss_d_b) * 0.5

    optimizer_d_a(grads_d_a)
    optimizer_d_b(grads_d_b)

    return loss_d

```

## Model Training

The training is divided into two main parts: training discriminator and training generator. In the discriminator loss function, the least-square loss is used to replace the negative log-likelihood objective.

- Training discriminator: The discriminator is trained to improve the probability of discriminating real images to the greatest extent. According to the method of the paper, the discriminator needs to be trained to minimize $E_{y-p_{data}(y)}[(D(y)-1)^2]$.

- Training generator: As described in the CycleGAN paper, we want to train the generator by minimizing $E_{x-p_{data}(x)}[(D(G(x)-1)^2]$ to produce better false images.

The following defines the training process of the generator and discriminator:

```python
import os
import time
import random
import numpy as np
from PIL import Image
from mindspore import Tensor, save_checkpoint
from mindspore import dtype

epochs = 7
save_step_num = 80
save_checkpoint_epochs = 1
save_ckpt_dir = './train_ckpt_outputs/'

print('Start training!')

iterator = dataset.create_dict_iterator(num_epochs=epochs)
for epoch in range(epochs):
    g_loss = []
    d_loss = []
    start_time_e = time.time()
    for step, data in enumerate(iterator):
        start_time_s = time.time()
        img_a = data["image_A"]
        img_b = data["image_B"]
        res_g = train_step_g(img_a, img_b)
        fake_a = res_g[0]
        fake_b = res_g[1]

        res_d = train_step_d(img_a, img_b, image_pool(fake_a), image_pool(fake_b))
        loss_d = float(res_d.asnumpy())
        step_time = time.time() - start_time_s

        res = []
        for item in res_g[2:]:
            res.append(float(item.asnumpy()))
        g_loss.append(res[0])
        d_loss.append(loss_d)

        if step % save_step_num == 0:
            print(f"Epoch:[{int(epoch + 1):>3d}/{int(epochs):>3d}], "
                  f"step:[{int(step):>4d}/{int(datasize):>4d}], "
                  f"time:{step_time:>3f}s,\n"
                  f"loss_g:{res[0]:.2f}, loss_d:{loss_d:.2f}, "
                  f"loss_g_a: {res[1]:.2f}, loss_g_b: {res[2]:.2f}, "
                  f"loss_c_a: {res[3]:.2f}, loss_c_b: {res[4]:.2f}, "
                  f"loss_idt_a: {res[5]:.2f}, loss_idt_b: {res[6]:.2f}")

    epoch_cost = time.time() - start_time_e
    per_step_time = epoch_cost / datasize
    mean_loss_d, mean_loss_g = sum(d_loss) / datasize, sum(g_loss) / datasize

    print(f"Epoch:[{int(epoch + 1):>3d}/{int(epochs):>3d}], "
          f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time:.2f}, "
          f"mean_g_loss:{mean_loss_g:.2f}, mean_d_loss:{mean_loss_d :.2f}")

    if epoch % save_checkpoint_epochs == 0:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        save_checkpoint(net_rg_a, os.path.join(save_ckpt_dir, f"g_a_{epoch}.ckpt"))
        save_checkpoint(net_rg_b, os.path.join(save_ckpt_dir, f"g_b_{epoch}.ckpt"))
        save_checkpoint(net_d_a, os.path.join(save_ckpt_dir, f"d_a_{epoch}.ckpt"))
        save_checkpoint(net_d_b, os.path.join(save_ckpt_dir, f"d_b_{epoch}.ckpt"))

print('End of training!')
```

```text

Start training!
Epoch:[  1/ 7], step:[   0/1019], time:6.202873s,
loss_g:22.88, loss_d:0.95, loss_g_a: 1.00, loss_g_b: 1.00, loss_c_a: 7.12, loss_c_b: 6.90, loss_idt_a: 3.52, loss_idt_b: 3.34
Epoch:[  1/ 7], step:[  80/1019], time:1.001927s,
loss_g:12.06, loss_d:0.49, loss_g_a: 0.51, loss_g_b: 0.26, loss_c_a: 3.98, loss_c_b: 3.76, loss_idt_a: 1.68, loss_idt_b: 1.87
Epoch:[  1/ 7], step:[ 160/1019], time:0.778982s,
loss_g:9.03, loss_d:0.43, loss_g_a: 0.68, loss_g_b: 0.61, loss_c_a: 2.20, loss_c_b: 2.99, loss_idt_a: 1.10, loss_idt_b: 1.45
Epoch:[  1/ 7], step:[ 240/1019], time:0.945285s,
loss_g:13.68, loss_d:0.33, loss_g_a: 0.54, loss_g_b: 0.39, loss_c_a: 4.33, loss_c_b: 4.61, loss_idt_a: 1.46, loss_idt_b: 2.35
Epoch:[  1/ 7], step:[ 320/1019], time:0.939093s,
...
Epoch:[  2/ 7], step:[ 960/1019], time:0.784652s,
loss_g:5.22, loss_d:0.52, loss_g_a: 0.23, loss_g_b: 0.34, loss_c_a: 1.83, loss_c_b: 1.59, loss_idt_a: 0.76, loss_idt_b: 0.47
Epoch:[  2/ 7], epoch time:923.28s, per step time:0.91, mean_g_loss:4.05, mean_d_loss:0.43,
Epoch:[  3/ 7], step:[   0/1019], time:0.744845s,
loss_g:3.74, loss_d:0.28, loss_g_a: 0.52, loss_g_b: 0.36, loss_c_a: 0.95, loss_c_b: 1.11, loss_idt_a: 0.40, loss_idt_b: 0.41
Epoch:[  3/ 7], step:[  80/1019], time:0.804595s,
loss_g:3.94, loss_d:0.71, loss_g_a: 0.18, loss_g_b: 0.32, loss_c_a: 1.40, loss_c_b: 0.97, loss_idt_a: 0.55, loss_idt_b: 0.51
Epoch:[  3/ 7], step:[ 160/1019], time:0.917811s,
loss_g:3.53, loss_d:0.68, loss_g_a: 0.30, loss_g_b: 0.17, loss_c_a: 1.23, loss_c_b: 0.91, loss_idt_a: 0.60, loss_idt_b: 0.32
Epoch:[  3/ 7], step:[ 240/1019], time:0.986027s,
loss_g:3.17, loss_d:0.49, loss_g_a: 0.34, loss_g_b: 0.18, loss_c_a: 0.75, loss_c_b: 1.15, loss_idt_a: 0.26, loss_idt_b: 0.50
...
Epoch:[  7/ 7], step:[ 720/1019], time:0.760178s,
loss_g:2.56, loss_d:0.40, loss_g_a: 0.43, loss_g_b: 0.37, loss_c_a: 0.47, loss_c_b: 0.70, loss_idt_a: 0.19, loss_idt_b: 0.42
Epoch:[  7/ 7], step:[ 800/1019], time:0.850483s,
loss_g:2.10, loss_d:0.61, loss_g_a: 0.33, loss_g_b: 0.13, loss_c_a: 0.58, loss_c_b: 0.55, loss_idt_a: 0.27, loss_idt_b: 0.24
Epoch:[  7/ 7], step:[ 880/1019], time:0.854865s,
loss_g:1.88, loss_d:0.55, loss_g_a: 0.37, loss_g_b: 0.14, loss_c_a: 0.45, loss_c_b: 0.43, loss_idt_a: 0.20, loss_idt_b: 0.29
Epoch:[  7/ 7], step:[ 960/1019], time:0.936919s,
loss_g:2.49, loss_d:0.54, loss_g_a: 0.38, loss_g_b: 0.14, loss_c_a: 0.66, loss_c_b: 0.53, loss_idt_a: 0.34, loss_idt_b: 0.44
Epoch:[  7/ 7], epoch time:911.41s, per step time:0.89, mean_g_loss:2.89, mean_d_loss:0.79,
End of training!
```

## Model Inference

Load the generator network model parameter file to migrate the style of the original image. In the result, the first row is the original image, and the second row is the generated result image.

```python
import os
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import load_checkpoint, load_param_into_net

# Load the weight file.
def load_ckpt(net, ckpt_dir):
    param_GA = load_checkpoint(ckpt_dir)
    load_param_into_net(net, param_GA)

g_a_ckpt = './CycleGAN_apple2orange/ckpt/g_a.ckpt'
g_b_ckpt = './CycleGAN_apple2orange/ckpt/g_b.ckpt'

load_ckpt(net_rg_a, g_a_ckpt)
load_ckpt(net_rg_b, g_b_ckpt)

# Image inference
fig = plt.figure(figsize=(11, 2.5), dpi=100)
def eval_data(dir_path, net, a):

    def read_img():
        for dir in os.listdir(dir_path):
            path = os.path.join(dir_path, dir)
            img = Image.open(path).convert('RGB')
            yield img, dir

    dataset = ds.GeneratorDataset(read_img, column_names=["image", "image_name"])
    trans = [vision.Resize((256, 256)), vision.Normalize(mean=[0.5 * 255] * 3, std=[0.5 * 255] * 3), vision.HWC2CHW()]
    dataset = dataset.map(operations=trans, input_columns=["image"])
    dataset = dataset.batch(1)
    for i, data in enumerate(dataset.create_dict_iterator()):
        img = data["image"]
        fake = net(img)
        fake = (fake[0] * 0.5 * 255 + 0.5 * 255).astype(np.uint8).transpose((1, 2, 0))
        img = (img[0] * 0.5 * 255 + 0.5 * 255).astype(np.uint8).transpose((1, 2, 0))

        fig.add_subplot(2, 8, i+1+a)
        plt.axis("off")
        plt.imshow(img.asnumpy())

        fig.add_subplot(2, 8, i+9+a)
        plt.axis("off")
        plt.imshow(fake.asnumpy())

eval_data('./CycleGAN_apple2orange/predict/apple', net_rg_a, 0)
eval_data('./CycleGAN_apple2orange/predict/orange', net_rg_b, 4)
plt.show()

```

## References

[1] I. Goodfellow. NIPS 2016 tutorial: Generative ad-versarial networks. arXiv preprint arXiv:1701.00160,2016. 2, 4, 5

[2] A. Shrivastava, T. Pfister, O. Tuzel, J. Susskind, W. Wang, R. Webb. Learning from simulated and unsupervised images through adversarial training. In CVPR, 2017. 3, 5, 6, 7
