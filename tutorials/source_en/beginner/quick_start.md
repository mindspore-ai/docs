# Quickstart: Handwritten Digit Recognition

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/quick_start.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This section runs through the basic process of MindSpore deep learning, using the LeNet5 network model as an example to implement common tasks in deep learning.

## Downloading and Processing the Dataset

Datasets are very important for model training, and good datasets can effectively improve training accuracy and efficiency. The MNIST dataset used in the example consists of 28∗28 grayscale images of 10 classes. The training dataset contains 60,000 images, and the test dataset contains 10,000 images.

![mnist](https://gitee.com/mindspore/docs/raw/master/tutorials/source_zh_cn/beginner/images/mnist.png)

> You can download it from the [MNIST dataset download page](http://yann.lecun.com/exdb/mnist/), unzip it and place it in the bottom directory structure.

The MindSpore Vision suite provides a Mnist module for downloading and processing MNIST datasets, and the following sample code downloads, extracts, and processes datasets to a specified location:

```python
from mindvision.dataset import Mnist

# Download and process the MNIST dataset
download_train = Mnist(path="./mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True)

download_eval = Mnist(path="./mnist", split="test", batch_size=32, resize=32, download=True)

dataset_train = download_train.run()
dataset_eval = download_eval.run()
```

Parameters description:

- path: dataset path.
- split: dataset type, supporting train, test, and infer, which defaults to train.
- batch_size: the data size set for each training batch, which defaults to 32.
- repeat_num: the number of times the dataset is traversed during training, which defaults to 1.
- shuffle: whether the dataset needs to be randomly scrambled (optional parameter).
- resize: the image size of the output image, which defaults to 32*32.
- download: whether you needs to download the dataset, which defaults to False.

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

According to the network structure of LeNet, there are 7 layers of LeNet removal input layer, including 3 convolutional layers, 2 sub-sampling layers, and 3 fully connected layers.

![](https://gitee.com/mindspore/docs/raw/master/tutorials/source_zh_cn/beginner/images/lenet.png)

The MindSpore Vision Suite provides the LeNet network model interface lenet, which defines the network model as follows:

```python
from mindvision.classification.models import lenet

network = lenet(num_classes=10, pretrained=False)
```

## Defining the Loss Function and the Optimizer

To train a neural network model, you need to define a loss function and an optimizer function.

- The loss function here uses the cross-entropy loss function `SoftmaxCrossEntropyWithLogits`.
- The optimizer here uses `Momentum`.

```python
import mindspore.nn as nn
from mindspore.train import Model

# Define the loss function
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# Define the optimizer function
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
```

## Training and Saving the Model

Before starting training, MindSpore needs to declare in advance whether the network model needs to save intermediate processes and results during training, so the `ModelCheckpoint` interface is used to save the network model and parameters for subsequent Fine-tuning operations.

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# Set the model saving parameter
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

# Apply the model saving parameter
ckpoint = ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)
```

The `model.train` interface provided by MindSpore makes it easy to train the network, and `LossMonitor` can monitor the change of `loss` value during training.

```python
from mindvision.engine.callback import LossMonitor

# Initialize the model parameter
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})

# Train the network model
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

The loss value will be printed during training, and the loss value will fluctuate, but in general, the loss value will gradually decrease and the accuracy will gradually increase. The loss values that each person runs have a certain randomness and are not necessarily exactly the same.

Verify the generalization capability of the model by running the test data set from the results obtained by running the model:

1. Use the `model.eval` interface to read in the test data set.
2. Use the saved model parameters for inference.

```python
acc = model.eval(dataset_eval)

print("{}".format(acc))
```

```text
{'accuracy': 0.9903846153846154}
```

The model accuracy data can be seen in the printed information. The accuracy data in the example reaches more than 95%, and the model quality is good. As the number of network iterations increases, the model accuracy increases further.

## Loading the Model

```python
from mindspore import load_checkpoint, load_param_into_net

# Load the model that has been saved for testing
param_dict = load_checkpoint("./lenet/lenet-1_1875.ckpt")
# Load parameters into the network
load_param_into_net(network, param_dict)
```

```text
[]
```

> For more information about loading a model in mindspore, see [Loading the Model](https://www.mindspore.cn/tutorials/en/master/save_load_model.html#loading-the-model).

## Validating the Model

Use the generated model to predict the classification of a single image. The procedure is as follows:

> The predicted images will be generated randomly, and the results may be different each time.

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

# Predict the image corresponding classification by using the function model.predict
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)

# Output prediction classification versus actual classification
print(f'Predicted: "{predicted}", Actual: "{labels}"')
```

![img](data:image/png;base64,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)

```text
Predicted: "[4 6 2 3 5 1]", Actual: "[4 6 2 3 5 1]"
```

As you can see from the printed results above, the predicted values are exactly the same as the target values.