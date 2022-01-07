# Using the VAE

<a href="https://gitee.com/mindspore/docs/blob/master/docs/probability/docs/source_en/using_the_vae.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

The following describes how to use the variational and dpn modules in MDP to implement VAE. VAE is a typical depth probabilistic model that applies variational inference to learn the representation of latent variables. The model can not only compress input data, but also generate new images of this type. The overall process is as follows:

1. Define a VAE.
2. Define the loss function and optimizer.
3. Process data.
4. Train the network.
5. Generate new samples or rebuild input samples.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/master/tests/st/probability/dpn>.

## Data Preparation

### Downloading the Dataset

In this example, using the MNIST_Data dataset, execute the following command to download and unzip it to the corresponding location:

```python
import os
import requests

def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)
```

```text
./datasets/MNIST_Data
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte

2 directories, 4 files
```

### Data Enhancement

The dataset is enhanced to meet the requirements of VAE network training. In this example, the pixel size of the original image is increased from $28\\times28$ to $32\\times32$, and multiple images are formed into a batch to accelerate training.

```python
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width))  # Bilinear mode
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.batch(batch_size)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds
```

### Defining the VAE

The composition of the variational autoencoder is mainly divided into three parts, the encoder, the decoder and the latent space. It is particularly simple to use the dpn module to construct a variational autoencoder. You only need to customize the encoder and decoder (DNN model) and call the VAE interface.

among them:

The main function of the Encoder is to reduce the dimensionality of the training data, compress, extract features, form a feature vector, and store it in the hidden space.

The main function of the decoder is to decode the parameters of the hidden space distribution of the training data and restore to generate a new image.

The main function of the hidden space is to store the characteristics of the model according to a certain distribution characteristic, which is a bridge between the encoder and the decoder.

```python
import os
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.nn.probability.dpn import VAE

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
IMAGE_SHAPE=(-1, 1, 32, 32)
image_path = os.path.join("./datasets/MNIST_Data", "train")

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

# define the encoder and decoder
encoder = Encoder()
decoder = Decoder()
# define the vae model
vae = VAE(encoder, decoder, hidden_size=400, latent_size=20)
```

### Defining the Loss Function and Optimizer

A loss function and an optimizer need to be defined. The loss function used in this example is `ELBO`, which is a loss function dedicated to variational inference. The optimizer used in this example is `nn.Adam`.
An example of the code for defining the loss function and optimizer in MindSpore is as follows:

```python
from mindspore.nn.probability.infer import ELBO

# define the loss function
net_loss = ELBO(latent_prior='Normal', output_prior='Normal')
# define the optimizer
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)
net_with_loss = nn.WithLossCell(vae, net_loss)
```

### Training the Generated Model

Generate training data, call the training mode of `vi` in the above code to train the model, and print out the loss value of the model after the training is completed.

```python
from mindspore.nn.probability.infer import SVI

vi = SVI(net_with_loss=net_with_loss,optimizer=optimizer)
# define the training dataset
ds_train = create_dataset(image_path, 32, 1)
# run the vi to return the trained network.
vae = vi.run(train_dataset=ds_train, epochs=10)
# get the trained loss
trained_loss = vi.get_train_loss()
print(trained_loss)
```

```text
45.09206426798502
```

### Building an Image Visualization Function

An image data with a batch of 32 can be visualized, which is convenient for comparing the difference between the original picture and the picture generated by the variational autoencoder.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_image(sample_data,col_num=4,row_num=8,count=0):
    for i in sample_data:
        plt.subplot(col_num,row_num,count+1)
        plt.imshow(np.squeeze(i.asnumpy()))
        plt.axis("off")
        count += 1
    plt.show()
```

### Randomly Generating Pictures

Use VAE random sampling to generate images.

```python
generated_sample = vae.generate_sample(32, IMAGE_SHAPE)

print("\n=============The Random generated Images=============")
plot_image(generated_sample)
```

### Rebuilding Input Samples

Use the trained model to check the ability to reconstruct the data. Here, take a set of original data for reconstruction and execute the following code:

```python
sample = next(ds_train.create_dict_iterator(output_numpy=True, num_epochs=1))
sample_x = Tensor(sample['image'], dtype=mstype.float32)

reconstructed_sample = vae.reconstruct_sample(sample_x)
print('The shape of the reconstructed sample is ', reconstructed_sample.shape)
print("\n=============The Original Images=============")
plot_image(sample_x)
print("\n============The Reconstruct Images=============")
plot_image(reconstructed_sample)
```