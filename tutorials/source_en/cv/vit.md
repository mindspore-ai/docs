[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/cv/vit.md)

# Vision Transformer Image Classification

Thanks to [ZOMI](https://gitee.com/sanjaychan) for contributing to this article.

## Introduction of ViT

In recent years, the development of natural language processing models has been greatly facilitated by the development of models based on Self-Attention (SAA) structures, especially the proposal of Transformer model. Due to its computational efficiency and scalability, Transformer has been able to train models of unprecedented size with over 100B parameters.

ViT is the convergence result of two fields: natural language processing and computer vision. It can still achieve good results on image classification tasks without relying on convolutional operations.

### Model Structure

The main structure of the ViT model is based on the Encoder part of the Transformer model (part of the structure order has been adjusted, e.g., the location of Normalization is different from that of the standard Transformer). Its structure diagram [1] is as follows:

![vit-architecture](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/vit_architecture.png)

### Model Features

ViT model is mainly applied in the field of image classification. Therefore, its model structure has the following features compared to the traditional Transformer:

1. After the original image of the dataset is divided into multiple patches, the two-dimensional patches (without considering the channel) are converted into one-dimensional vectors. The one-dimensional vectors, the category vectors and the position vectors are added as model inputs.
2. The Block structure of the main body in the model is based on the Encoder structure of Transformer, but adjusts the position of Normalization, where the main structure is still the Multi-head Attention structure.
3. The model connects a fully connected layer after the Blocks stack, accepts the output of the category vector as input and uses it for classification. Typically, we refer to the final fully-onnected layer as Head and the Transformer Encoder part as backbone.

The following code example will explain in detail the implementation of ImageNet classification task based on ViT.

> This tutorial runs on the CPU for too long and is not recommended to run on the CPU.

## Environment Preparation and Data Reading

Before you start experimenting, make sure you have installed a local Python environment and MindSpore.

First we need to download the dataset for this case. The complete ImageNet dataset can be downloaded via <http://image-net.org>. The dataset applied in this case is a subset filtered out from ImageNet.

The first code will be downloaded and unpacked automatically when you run it. Make sure your dataset path is structured as follows.

```text
.dataset/
    ├── ILSVRC2012_devkit_t12.tar.gz
    ├── train/
    ├── infer/
    └── val/
```

```python
from download import download

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/vit_imagenet_dataset.zip"
path = "./"

path = download(dataset_url, path, kind="zip", replace=True)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/vit_imagenet_dataset.zip (489.1 MB)

file_sizes: 100%|████████████████████████████| 513M/513M [00:09<00:00, 52.3MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

```python
import os

import mindspore as ms
from mindspore.dataset import ImageFolderDataset
import mindspore.dataset.vision as transforms


data_path = './dataset/'
mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

dataset_train = ImageFolderDataset(os.path.join(data_path, "train"), shuffle=True)

trans_train = [
    transforms.RandomCropDecodeResize(size=224,
                                      scale=(0.08, 1.0),
                                      ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.Normalize(mean=mean, std=std),
    transforms.HWC2CHW()
]

dataset_train = dataset_train.map(operations=trans_train, input_columns=["image"])
dataset_train = dataset_train.batch(batch_size=16, drop_remainder=True)
```

## Model Analysis

The following is a detailed dissection of the internal structure of the ViT model through the code.

### Basic Principle of Transformer

The Transformer model originated from a 2017 article [2]. The encoder-decoder type structure based on the Attention mechanism proposed in this article has been a great success in the field of natural language processing. The model structure is shown in the following figure:

![transformer-architecture](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/transformer_architecture.png)

Its main structure is composed of several Encoder and Decoder modules, where the detailed structure of Encoder and Decoder is shown in the following figure [2]:

![encoder-decoder](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/encoder_decoder.png)

Encoder and Decoder consist of many structures, such as Multi-Head Attention layer, Feed Forward layer, Normaliztion layer, and even Residual Connection ("Add" in the figure). However, one of the most important structures is the Multi-Head Attention structure, which is based on the Self-Attention mechanism and is a parallel composition of multiple Self-Attentions.

Therefore, understanding Self-Attention means understanding the core of Transformer.

#### Attention Module

The following is an explanation of Self-Attention, the core of which is to learn a weight for each word of the input vector. Given a task-related query vector Query vector, the similarity or relevance of Query and each Key is calculated to obtain the Attention distribution, i.e., the weight coefficient of each Key corresponding to Value is obtained, and then the Value is weighted and summed to obtain the final Attention value.

In the Self-Attention:

1. The initial input vector is first mapped into three vectors Q(Query), K(Key), and V(Value) by the Embedding layer, and since it is a parallel operation, the code is mapped into a dim x 3 vector and partitioned. In other words, if your input vector is a sequence of vectors ($x_1$, $x_2$, $x_3$), where $ x_1$, $x_2$, $x_3$ are one-dimensional vectors, each one-dimensional vector is mapped to Q, K, and V by the Embedding layer, only the Embedding matrix is different and the matrix parameters are obtained through learning. **Here we can consider that the three matrices Q, K, V are a means to discover the correlation information between vectors, which needs to be obtained through learning. The reason for the number of vector is three is that two vectors can get the weights after point multiplication, and another vector is needed to carry the results of summing the weights, so at least three matrices are needed.**

    $$
    \begin{cases}
    q_i = W_q \cdot x_i & \\
    k_i = W_k \cdot x_i,\hspace{1em} &i = 1,2,3 \ldots \\
    v_i = W_v \cdot x_i &
    \end{cases}
    \tag{1}
    $$

    ![self-attention1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/self_attention_1.png)

2. The self-attentiveness of the self-attentive mechanism is mainly reflected by the fact that its Q, K, and V all originate from itself, that is, the process is extracting the connections and features of the input vectors of different orders, which are finally expressed by the connection closeness between the vectors of different orders (the result of the product of Q and K after Softmax). **After obtaining Q, K, V, we need to obtain the inter-vector weights, that is, to point multiple Q and K and divide by the square root of the dimension, and Softmax the results of all vectors. By the operation in equation (2), we obtain the relation weights between vectors.**

    $$
    \begin{cases}
    a_{1,1} = q_1 \cdot k_1 / \sqrt d \\
    a_{1,2} = q_1 \cdot k_2 / \sqrt d \\
    a_{1,3} = q_1 \cdot k_3 / \sqrt d
    \end{cases}
    \tag{2}
    $$

    ![self-attention3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/self_attention_3.png)

    $$ Softmax: \hat a_{1,i} = exp(a_{1,i}) / \sum_j exp(a_{1,j}),\hspace{1em} j = 1,2,3 \ldots \tag{3}$$

    ![self-attention2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/self_attention_2.png)

3. The final output is obtained by weight sum of the mapped vector V with Q, K after Softmax, and the process can be understood as a global self-attentive representation. **Each set of Q, K, and V ends up with a V output, which is the final result obtained by Self-Attention, and is the result of the current vector after combining its associated weights with other vectors.**

    $$
    b_1 = \sum_i \hat a_{1,i}v_i,\hspace{1em} i = 1,2,3...
    \tag{4}
    $$

The following diagram provides an overall grasp of the entire Self-Attention process.

![self-attention](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/self_attention_process.png)

The multi-head attention mechanism is to split the vector originally processed by self-Attention into multiple Heads for processing, which can also be reflected in the code, which is one aspect of the attention structure that allows parallel acceleration.

To summarize, the multi-head attention mechanism maps the same query, key and value to different subspaces (Q_0,K_0,V_0) of the original high-dimensional space (Q,K,V) for self-attention computation while keeping the total number of parameters unchanged, and finally merges the attention information in different subspaces.

Therefore, for the same input vector, multiple attention mechanisms can process it simultaneously, i.e., using parallel computing to speed up the processing process and analyzing and utilizing the vector features during the processing. The following figure shows the multi-headed attention mechanism, whose parallelism capability is mainly reflected by the fact that $a_1$ and $a_2$ are obtained by partitioning the same vector in the following figure.

![multi-head-attention](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/multi_head_attention.png)

The following Multi-Head Attention code, combined with the explanation above, clearly shows the process.

```python
from mindspore import nn, ops


class Attention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ms.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(p=1.0-attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(p=1.0-keep_prob)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out
```

### Transformer Encoder

After understanding the Self-Attention structure, the basic structure of Transformer can be formed by splicing with Feed Forward, Residual Connection, etc. The following code implements structures of Feed Forward and Residual Connection.

```python
from typing import Optional, Dict


class FeedForward(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(p=1.0-keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


class ResidualCell(nn.Cell):
    def __init__(self, cell):
        super(ResidualCell, self).__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x
```

Next, Self-Attention is used to construct the TransformerEncoder part in the ViT model, similar to constructing the encoder part of a Transformer, as shown in the following figure [1]:

![vit-encoder](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/vit_encoder.png)

1. The basic structure in the ViT model is different from that of the standard Transformer, mainly in that the position of Normalization is placed before Self-Attention and Feed Forward, while other structures such as Residual Connection, Feed Forward, and Normalization are designed as the structure in the Transformer.

2. From the picture of Transformer structure, we can find that the stacking of multiple sub-encoders completes the construction of the model encoder. In the ViT model, this idea is still followed, and the number of stacking layers can be determined by configuring the hyperparameter num_layers.

3. The structures of Residual Connection and Normalization can ensure that the model has strong scalability (to ensure that the information will not degrade after deep processing, which is the role of Residual Connection), and the application of Normalization and dropout can enhance the model generalization ability.

The Transformer structure can be clearly seen in the following source code. Combining the TransformerEncoder structure with a multilayer perceptron (MLP) forms the backbone part in the ViT model.

```python
class TransformerEncoder(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerEncoder, self).__init__()
        layers = []

        for _ in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  keep_prob=keep_prob,
                                  attention_keep_prob=attention_keep_prob)

            feedforward = FeedForward(in_features=dim,
                                      hidden_features=mlp_dim,
                                      activation=activation,
                                      keep_prob=keep_prob)

            layers.append(
                nn.SequentialCell([
                    ResidualCell(nn.SequentialCell([normalization1, attention])),
                    ResidualCell(nn.SequentialCell([normalization2, feedforward]))
                ])
            )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Transformer construct."""
        return self.layers(x)
```

### Input for ViT model

The traditional Transformer structure is mainly used to deal with Word Embedding or Word Vector in the natural language domain. The main difference between word vectors and traditional image data is that word vectors are usually 1-dimensional vectors for stacking, while images are stacked in a 2-dimensional matrix. The multi-headed attention mechanism extracts the connection between word vectors, i.e., the contextual semantics, when dealing with the stack of 1-dimensional word vectors, which makes Transformer very useful in the field of natural language processing. How the 2-dimensional image matrix is transformed with 1-dimensional word vectors becomes a small threshold for Transformer to enter the field of image processing.

In the ViT model:

1. By dividing the input image into 16 x 16 patches on each channel, this step is done by a convolution operation, which can of course be done manually. However, convolutional operations can also serve the purpose while still allowing for additional data processing. **For example, an input 224 x 224 image is first convolved to get 14 x 14 patches, so the size of each patch is 16 x 16.**

2. The matrix of each patch is then stretched into a 1-dimensional vector, thus obtaining an approximate word vector stacking effect. **The series of patches of size 16 x 16 obtained in the previous step is then converted to a vector of length 196.**

This is the first processing step of the image input network. The code for the specific Patch Embedding is shown below:

```python
class PatchEmbedding(nn.Cell):
    MIN_NUM_PATCHES = 4

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 input_channels: int = 3):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def construct(self, x):
        """Path Embedding construct."""
        x = self.conv(x)
        b, c, h, w = x.shape
        x = ops.reshape(x, (b, c, h * w))
        x = ops.transpose(x, (0, 2, 1))

        return x
```

After the input image is divided into patches, it goes through two processes, pos_embedding and class_embedding.

1. class_embedding mainly borrows from the BERT model for text classification, adding a class value before each word vector, usually in the first position of the vector. **The 196-dimensional vector obtained in the previous step becomes 197-dimension after adding class_embedding.**

2. The added class_embedding is a parameter that can be learned. After continuous training of the network, the output of the first dimension in the output vector is ultimately used to determine the final output class. **Since the input is 14 x 14 patch, the output is classified by taking 14 x 14 class_embeddings for classification.**

3. The pos_embedding is also a set of parameters that can be learned and will be added to the processed patch matrix.

4. Since pos_embedding is also a parameter that can be learned, it is joined similarly to the bias of fully-linked networks and convolution. **This step is to create a trainable vector of 197-dimension to be added to the vector that has gone through class_embedding.**

In fact, there are a total of 4 options for pos_embedding. But after the authors' arguments, only adding pos_embedding and not adding pos_embedding has a significant impact. Whether pos_embedding is 1-dimensional or 2-dimensional has little effect on the classification result, so in code, the 1-dimensional pos_embedding is also applied. Since class_embedding is added before pos_embedding, the dimension of pos_embedding will be plus 1 than the dimension after patch stretching.

In general, the ViT model takes advantage of the Transformer model in processing contextual semantics by converting images into a "variant word vector" and then processing them. The significance of this conversion is that multiple patches are spatially connected, which is similar to a "spatial semantics", and thus achieves better processing results.

### Building ViT as a whole

The following code constructs a complete ViT model.

```python
from mindspore.common.initializer import Normal
from mindspore.common.initializer import initializer
from mindspore import Parameter


def init(init_type, shape, dtype, name, requires_grad):
    """Init."""
    initial = initializer(init_type, shape, dtype).init_data()
    return Parameter(initial, name=name, requires_grad=requires_grad)


class ViT(nn.Cell):
    def __init__(self,
                 image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: Optional[nn.Cell] = nn.LayerNorm,
                 pool: str = 'cls') -> None:
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=input_channels)
        num_patches = self.patch_embedding.num_patches

        self.cls_token = init(init_type=Normal(sigma=1.0),
                              shape=(1, 1, embed_dim),
                              dtype=ms.float32,
                              name='cls',
                              requires_grad=True)

        self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                  shape=(1, num_patches + 1, embed_dim),
                                  dtype=ms.float32,
                                  name='pos_embedding',
                                  requires_grad=True)

        self.pool = pool
        self.pos_dropout = nn.Dropout(p=1.0-keep_prob)
        self.norm = norm((embed_dim,))
        self.transformer = TransformerEncoder(dim=embed_dim,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              keep_prob=keep_prob,
                                              attention_keep_prob=attention_keep_prob,
                                              drop_path_keep_prob=drop_path_keep_prob,
                                              activation=activation,
                                              norm=norm)
        self.dropout = nn.Dropout(p=1.0-keep_prob)
        self.dense = nn.Dense(embed_dim, num_classes)

    def construct(self, x):
        """ViT construct."""
        x = self.patch_embedding(x)
        cls_tokens = ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
        x = ops.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding

        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]
        if self.training:
            x = self.dropout(x)
        x = self.dense(x)

        return x
```

The overall flow diagram is shown below:

![data-process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/cv/images/data_process.png)

## Model Training and Inference

### Model Training

Before the model starts training, the loss function, optimizer, and callback function need to be set.

It takes a long time to train the ViT model completely, and it is recommended to adjust the epoch_size according to the needs of the project when it is actually applied. When the normal output of the step information in each Epoch means that the training is in progress, the model output can be used to view loss value, time and other indicators of the current training.

```python
from mindspore.nn import LossBase
from mindspore.train import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import train

# define super parameter
epoch_size = 10
momentum = 0.9
num_classes = 1000
resize = 224
step_size = dataset_train.get_dataset_size()

# construct model
network = ViT()

# load ckpt
vit_url = "https://download.mindspore.cn/vision/classification/vit_b_16_224.ckpt"
path = "./ckpt/vit_b_16_224.ckpt"

vit_path = download(vit_url, path, replace=True)
param_dict = ms.load_checkpoint(vit_path)
ms.load_param_into_net(network, param_dict)

# define learning rate
lr = nn.cosine_decay_lr(min_lr=float(0),
                        max_lr=0.00005,
                        total_step=epoch_size * step_size,
                        step_per_epoch=step_size,
                        decay_epoch=10)

# define optimizer
network_opt = nn.Adam(network.trainable_params(), lr, momentum)


# define loss function
class CrossEntropySmooth(LossBase):
    """CrossEntropy."""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss


network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)

# set checkpoint
ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=100)
ckpt_callback = ModelCheckpoint(prefix='vit_b_16', directory='./ViT', config=ckpt_config)

# initialize model
# "Ascend + mixed precision" can improve performance
ascend_target = (ms.get_context("device_target") == "Ascend")
if ascend_target:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O2")
else:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O0")

# train model
model.train(epoch_size,
            dataset_train,
            callbacks=[ckpt_callback, LossMonitor(125), TimeMonitor(125)],
            dataset_sink_mode=False,)
```

```text
Downloading data from https://download.mindspore.cn/vision/classification/vit_b_16_224.ckpt (330.2 MB)

file_sizes: 100%|████████████████████████████| 346M/346M [00:05<00:00, 59.5MB/s]
Successfully downloaded file to ./ckpt/vit_b_16_224.ckpt
epoch: 1 step: 125, loss is 1.903618335723877
Train epoch time: 99857.517 ms, per step time: 798.860 ms
epoch: 2 step: 125, loss is 1.448015570640564
Train epoch time: 95555.111 ms, per step time: 764.441 ms
epoch: 3 step: 125, loss is 1.2555729150772095
Train epoch time: 95553.210 ms, per step time: 764.426 ms
epoch: 4 step: 125, loss is 1.3787992000579834
Train epoch time: 95569.835 ms, per step time: 764.559 ms
epoch: 5 step: 125, loss is 1.7505667209625244
Train epoch time: 95463.133 ms, per step time: 763.705 ms
epoch: 6 step: 125, loss is 2.5462236404418945
Train epoch time: 95428.906 ms, per step time: 763.431 ms
epoch: 7 step: 125, loss is 1.5103509426116943
Train epoch time: 95411.338 ms, per step time: 763.291 ms
epoch: 8 step: 125, loss is 1.4719784259796143
Train epoch time: 95644.054 ms, per step time: 765.152 ms
epoch: 9 step: 125, loss is 1.2415032386779785
Train epoch time: 95511.758 ms, per step time: 764.094 ms
epoch: 10 step: 125, loss is 1.098097562789917
Train epoch time: 95270.282 ms, per step time: 762.162 ms
```

### Model Validation

The model validation process mainly applies interfaces such as [ImageFolderDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.ImageFolderDataset.html), CrossEntropySmooth and Model.

ImageFolderDataset is mainly used to read datasets.

CrossEntropySmooth is the loss function instantiation interface.

Model is mainly used to compile models.

Similar to the training process, data augmentation is first performed, then the ViT network structure is defined and pre-trained model parameters are loaded. Subsequently, the loss function and evaluation metrics are set, and the model is compiled and then validated. This case uses the Top_1_Accuracy and Top_5_Accuracy evaluation metrics, which are commonly used in the industry, to evaluate the model performance.

In this case, these two metrics represent the accuracy of the model prediction when the categories represented by the maximum or the top 5 output values are the prediction results in the output 1000-dimensional vector. Larger values of these two metrics represent higher model accuracy.

```python
dataset_val = ImageFolderDataset(os.path.join(data_path, "val"), shuffle=True)

trans_val = [
    transforms.Decode(),
    transforms.Resize(224 + 32),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=mean, std=std),
    transforms.HWC2CHW()
]

dataset_val = dataset_val.map(operations=trans_val, input_columns=["image"])
dataset_val = dataset_val.batch(batch_size=16, drop_remainder=True)

# construct model
network = ViT()

# load ckpt
param_dict = ms.load_checkpoint(vit_path)
ms.load_param_into_net(network, param_dict)

network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=num_classes)

# define metric
eval_metrics = {'Top_1_Accuracy': train.Top1CategoricalAccuracy(),
                'Top_5_Accuracy': train.Top5CategoricalAccuracy()}

if ascend_target:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O2")
else:
    model = train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O0")

# evaluate model
result = model.eval(dataset_val)
print(result)
```

```text
{'Top_1_Accuracy': 0.75, 'Top_5_Accuracy': 0.928}
```

From the results, we can see that the Top_1_Accuracy and Top_5_Accuracy of the model reach a high level due to our loading of pre-trained model parameters, and this accuracy can be used as the standard in the actual project. If the pre-trained model parameters are not used, more epochs are needed for training.

### Model Inference

Before performing model inference, a method for preprocessing data on the inferred images is first defined. This method allows us to resize and normalize our inference images so that they can match our input data during training.

In this case, a picture of Doberman is used as an inference picture to test the model performance, and the model is expected to give correct predicted results.

```python
dataset_infer = ImageFolderDataset(os.path.join(data_path, "infer"), shuffle=True)

trans_infer = [
    transforms.Decode(),
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=mean, std=std),
    transforms.HWC2CHW()
]

dataset_infer = dataset_infer.map(operations=trans_infer,
                                  input_columns=["image"],
                                  num_parallel_workers=1)
dataset_infer = dataset_infer.batch(1)
```

Next, we will call the predict method of the model for the model inference.

In the inference process, the corresponding label can be obtained by index2label, and then the result is written on the corresponding image through the custom show_result interface.

```python
import os
import pathlib
import cv2
import numpy as np
from PIL import Image
from enum import Enum
from scipy import io


class Color(Enum):
    """dedine enum color."""
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def check_file_exist(file_name: str):
    """check_file_exist."""
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File `{file_name}` does not exist.")


def color_val(color):
    """color_val."""
    if isinstance(color, str):
        return Color[color].value
    if isinstance(color, Color):
        return color.value
    if isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    if isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    if isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    raise TypeError(f'Invalid type for color: {type(color)}')


def imread(image, mode=None):
    """imread."""
    if isinstance(image, pathlib.Path):
        image = str(image)

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        check_file_exist(image)
        image = Image.open(image)
        if mode:
            image = np.array(image.convert(mode))
    else:
        raise TypeError("Image must be a `ndarray`, `str` or Path object.")

    return image


def imwrite(image, image_path, auto_mkdir=True):
    """imwrite."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(image_path))
        if dir_name != '':
            dir_name = os.path.expanduser(dir_name)
            os.makedirs(dir_name, mode=777, exist_ok=True)

    image = Image.fromarray(image)
    image.save(image_path)


def imshow(img, win_name='', wait_time=0):
    """imshow"""
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def show_result(img: str,
                result: Dict[int, float],
                text_color: str = 'green',
                font_scale: float = 0.5,
                row_width: int = 20,
                show: bool = False,
                win_name: str = '',
                wait_time: int = 0,
                out_file: Optional[str] = None) -> None:
    """Mark the prediction results on the picture."""
    img = imread(img, mode="RGB")
    img = img.copy()
    x, y = 0, row_width
    text_color = color_val(text_color)
    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, text_color)
        y += row_width
    if out_file:
        show = False
        imwrite(img, out_file)

    if show:
        imshow(img, win_name, wait_time)


def index2label():
    """Dictionary output for image numbers and categories of the ImageNet dataset."""
    metafile = os.path.join(data_path, "ILSVRC2012_devkit_t12/data/meta.mat")
    meta = io.loadmat(metafile, squeeze_me=True)['synsets']

    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]

    _, wnids, classes = list(zip(*meta))[:3]
    clssname = [tuple(clss.split(', ')) for clss in classes]
    wnid2class = {wnid: clss for wnid, clss in zip(wnids, clssname)}
    wind2class_name = sorted(wnid2class.items(), key=lambda x: x[0])

    mapping = {}
    for index, (_, class_name) in enumerate(wind2class_name):
        mapping[index] = class_name[0]
    return mapping


# Read data for inference
for i, image in enumerate(dataset_infer.create_dict_iterator(output_numpy=True)):
    image = image["image"]
    image = ms.Tensor(image)
    prob = model.predict(image)
    label = np.argmax(prob.asnumpy(), axis=1)
    mapping = index2label()
    output = {int(label): mapping[int(label)]}
    print(output)
    show_result(img="./dataset/infer/n01440764/ILSVRC2012_test_00000279.JPEG",
                result=output,
                out_file="./dataset/infer/ILSVRC2012_test_00000279.JPEG")
```

```text
{236: 'Doberman'}
```

After the inference process is completed, the inference result of the picture can be found under the inference folder, and it can be seen that the prediction result is Doberman, which is the same as the expected result and verifies the accuracy of the model.

![infer-result](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/cv/images/infer_result.jpg)

## Summary

This case completes the process of training, validation and inference of a ViT model on ImageNet data, in which the key ViT model structure and principles are explained. By studying this case, understanding the source code can help users to master key concepts such as Multi-Head Attention, TransformerEncoder, pos_embedding, etc. If you want to understand ViT's model principles in detail, it is recommended to read it in deeper detail based on the source code.
