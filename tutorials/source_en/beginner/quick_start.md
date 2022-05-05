# Quickstart: Handwritten Digit Recognition

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/source_en/beginner/quick_start.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

This section runs through the basic process of MindSpore deep learning, using the LeNet5 network model as an example to implement common tasks in deep learning.

## Downloading and Processing the Dataset

Datasets are crucial for model training. A good dataset can effectively improve training accuracy and efficiency. The MNIST dataset used in the example consists of 28 x 28 grayscale images of 10 classes. The training dataset contains 60,000 images, and the test dataset contains 10,000 images.

![mnist](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/mnist.png)

> You can download the dataset from [MNIST dataset download page](http://yann.lecun.com/exdb/mnist/), decompress it, and save it according to the following directory structure.

The [MindSpore Vision](https://mindspore.cn/vision/docs/en/r0.1/index.html) suite provides the Mnist module for downloading and processing MNIST datasets. The following sample code downloads and decompresses the datasets to the specified location for data processing:

```python
from mindvision.dataset import Mnist

# Download and process the MNIST dataset.
download_train = Mnist(path="./mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True)

download_eval = Mnist(path="./mnist", split="test", batch_size=32, resize=32, download=True)

dataset_train = download_train.run()
dataset_eval = download_eval.run()
```

Parameters description:

- path: dataset path.
- split: dataset type. The value can be train, test, or infer. The default value is train.
- batch_size: data size set for each training batch. The default value is 32.
- repeat_num: number of times that the dataset is traversed during training. The default value is 1.
- shuffle: determines whether to randomly shuffle the dataset. This parameter is optional.
- resize: size of the output image. The default value is 32 x 32.
- download: determines whether to download the dataset. The default value is False.

The directory structure of the downloaded dataset files is as follows:

```text
./mnist/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

## Building the Model

Except the input layer, LeNet contains seven layers: three convolutional layers, two subsampling layers, and two fully-connected layers.

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/lenet.png)

The MindSpore Vision Suite provides the LeNet model interface `lenet`, which defines the network model as follows:

```python
from mindvision.classification.models import lenet

network = lenet(num_classes=10, pretrained=False)
```

## Defining a Loss Function and an Optimizer

To train a neural network model, you need to define a loss function and an optimizer function.

- The following uses the cross-entropy loss function `SoftmaxCrossEntropyWithLogits`.
- The optimizer is `Momentum`.

```python
import mindspore.nn as nn
from mindspore.train import Model

# Define the loss function.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# Define the optimizer function.
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
```

## Training and Saving the Model

Before training, MindSpore needs to declare whether the intermediate process and result of the network model need to be saved during training. Therefore, the `ModelCheckpoint` API is used to save the network model and parameters for subsequent fine-tuning.

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# Set the model saving parameters.
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

# Apply the model saving parameters.
ckpoint = ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)
```

The `model.train` API provided by MindSpore can be used to easily train the network. `LossMonitor` can monitor the changes of the `loss` value during the training process.

```python
from mindvision.engine.callback import LossMonitor

# Initialize the model parameters.
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})

# Train the network model.
model.train(10, dataset_train, callbacks=[ckpoint, LossMonitor(0.01, 1875)])
```

```text
Epoch:[  0/ 10], step:[ 1875/ 1875], loss:[0.314/0.314], time:2237.313 ms, lr:0.01000
Epoch time: 3577.754 ms, per step time: 1.908 ms, avg loss: 0.314
Epoch:[  1/ 10], step:[ 1875/ 1875], loss:[0.031/0.031], time:1306.982 ms, lr:0.01000
Epoch time: 1307.792 ms, per step time: 0.697 ms, avg loss: 0.031
Epoch:[  2/ 10], step:[ 1875/ 1875], loss:[0.007/0.007], time:1324.625 ms, lr:0.01000
Epoch time: 1325.340 ms, per step time: 0.707 ms, avg loss: 0.007
Epoch:[  3/ 10], step:[ 1875/ 1875], loss:[0.021/0.021], time:1396.733 ms, lr:0.01000
Epoch time: 1397.495 ms, per step time: 0.745 ms, avg loss: 0.021
Epoch:[  4/ 10], step:[ 1875/ 1875], loss:[0.028/0.028], time:1594.762 ms, lr:0.01000
Epoch time: 1595.549 ms, per step time: 0.851 ms, avg loss: 0.028
Epoch:[  5/ 10], step:[ 1875/ 1875], loss:[0.007/0.007], time:1242.175 ms, lr:0.01000
Epoch time: 1242.928 ms, per step time: 0.663 ms, avg loss: 0.007
Epoch:[  6/ 10], step:[ 1875/ 1875], loss:[0.033/0.033], time:1199.938 ms, lr:0.01000
Epoch time: 1200.627 ms, per step time: 0.640 ms, avg loss: 0.033
Epoch:[  7/ 10], step:[ 1875/ 1875], loss:[0.175/0.175], time:1228.845 ms, lr:0.01000
Epoch time: 1229.548 ms, per step time: 0.656 ms, avg loss: 0.175
Epoch:[  8/ 10], step:[ 1875/ 1875], loss:[0.009/0.009], time:1237.200 ms, lr:0.01000
Epoch time: 1237.969 ms, per step time: 0.660 ms, avg loss: 0.009
Epoch:[  9/ 10], step:[ 1875/ 1875], loss:[0.000/0.000], time:1287.693 ms, lr:0.01000
Epoch time: 1288.413 ms, per step time: 0.687 ms, avg loss: 0.000
```

During the training, the loss value is printed and fluctuates. However, the loss value gradually decreases and the precision gradually increases. Loss values displayed each time may be different because of their randomness.

Validate the generalization capability of the model based on the result obtained by running the test dataset.

1. Read the test dataset using the `model.eval` API.
2. Use the saved model parameters for inference.

```python
acc = model.eval(dataset_eval)

print("{}".format(acc))
```

```text
{'accuracy': 0.9903846153846154}
```

The model accuracy data is displayed in the output content. In the example, the accuracy reaches 95%, indicating a good model quality. As the number of network epochs increases, the model accuracy will be further improved.

## Loading the Model

```python
from mindspore import load_checkpoint, load_param_into_net

# Load the saved model used for testing.
param_dict = load_checkpoint("./lenet/lenet-1_1875.ckpt")
# Load parameters to the network.
load_param_into_net(network, param_dict)
```

```text
[]
```

> For more information about loading a model in mindspore, see [Loading the Model](https://www.mindspore.cn/tutorials/en/r1.7/beginner/save_load.html#loading-the-model).

## Validating the Model

Use the generated model to predict the classification of a single image. The procedure is as follows:

> The predicted image is randomly generated, and the execution result may be different each time.

```python
import numpy as np
from mindspore import Tensor
import matplotlib.pyplot as plt

mnist = Mnist("./mnist", split="train", batch_size=6, resize=32)
dataset_infer = mnist.run()
ds_test = dataset_infer.create_dict_iterator()
data = next(ds_test)
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

plt.figure()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.imshow(images[i-1][0], interpolation="None", cmap="gray")
plt.show()

# Use the model.predict function to predict the classification of the image.
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)

# Output the predicted classification and the actual classification.
print(f'Predicted: "{predicted}", Actual: "{labels}"')
```

```text
Predicted: "[4 6 2 3 5 1]", Actual: "[4 6 2 3 5 1]"
```

The preceding information shows that the predicted values are the same as the target values.
