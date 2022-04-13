# Saving and Loading the Model

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/save_load.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The content of the previous chapter mainly introduced how to adjust hyperparameters and train network models. During the process of training the network model, we actually want to save the intermediate and final results for fine-tune and subsequent model deployment and inference, and now start learning how to set hyperparameters and optimize model parameters.

## Model Training

The following are the basic steps and code for network model training, with sample code as follows:

```python
import mindspore.nn as nn
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor

epochs = 10 # Training batch

# 1. Build a dataset
download_train = Mnist(path="./mnist", split="train", batch_size=batch_size, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

# 2. Define a neural network
network = lenet(num_classes=10, pretrained=False)
# 3.1 Define a loss function
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 3.2 Defines an optimizer function
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
# 3.3 Initialize model parameters
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})

# 4. Perform training on the neural network
model.train(epochs, dataset_train, callbacks=[LossMonitor(0.01, 1875)])
```

```text
Epoch:[  0/ 10], step:[ 1875/ 1875], loss:[0.148/1.210], time:2.021 ms, lr:0.01000
Epoch time: 4251.808 ms, per step time: 2.268 ms, avg loss: 1.210
Epoch:[  1/ 10], step:[ 1875/ 1875], loss:[0.049/0.081], time:2.048 ms, lr:0.01000
Epoch time: 4301.405 ms, per step time: 2.294 ms, avg loss: 0.081
Epoch:[  2/ 10], step:[ 1875/ 1875], loss:[0.014/0.050], time:1.992 ms, lr:0.01000
Epoch time: 4278.799 ms, per step time: 2.282 ms, avg loss: 0.050
Epoch:[  3/ 10], step:[ 1875/ 1875], loss:[0.035/0.038], time:2.254 ms, lr:0.01000
Epoch time: 4380.553 ms, per step time: 2.336 ms, avg loss: 0.038
Epoch:[  4/ 10], step:[ 1875/ 1875], loss:[0.130/0.031], time:1.932 ms, lr:0.01000
Epoch time: 4287.547 ms, per step time: 2.287 ms, avg loss: 0.031
Epoch:[  5/ 10], step:[ 1875/ 1875], loss:[0.003/0.027], time:1.981 ms, lr:0.01000
Epoch time: 4377.000 ms, per step time: 2.334 ms, avg loss: 0.027
Epoch:[  6/ 10], step:[ 1875/ 1875], loss:[0.004/0.023], time:2.167 ms, lr:0.01000
Epoch time: 4687.250 ms, per step time: 2.500 ms, avg loss: 0.023
Epoch:[  7/ 10], step:[ 1875/ 1875], loss:[0.004/0.020], time:2.226 ms, lr:0.01000
Epoch time: 4685.529 ms, per step time: 2.499 ms, avg loss: 0.020
Epoch:[  8/ 10], step:[ 1875/ 1875], loss:[0.000/0.016], time:2.275 ms, lr:0.01000
Epoch time: 4651.129 ms, per step time: 2.481 ms, avg loss: 0.016
Epoch:[  9/ 10], step:[ 1875/ 1875], loss:[0.022/0.015], time:2.177 ms, lr:0.01000
Epoch time: 4623.760 ms, per step time: 2.466 ms, avg loss: 0.015
```

As you can see from the printed results above, the loss values tend to converge as the number of training rounds increases.

## Saving the Model

After training the network, the following will describe how to save and load the model. There are two main ways to save the interface of the model:

1. One is to simply save the network model, which can be saved before and after training. The advantage is that the interface is simple and easy to use, but only the state of the network model when the command is executed is retained;

2. The other one is to save the interface during network model training. In the process of network model training, MindSpore automatically saves the parameters of the epoch number and step number set during training, that is, the intermediate weight parameters generated during the model training process are also saved to facilitate network fine-tuning and stop training.

### Saving the Model Directly

Use the save_checkpoint provided by MindSpore to save the model, and pass it to the network and save the path:

```python
import mindspore as ms

# The defined network model is net, which is generally used before or after training
ms.save_checkpoint(net, "./MyNet.ckpt")
```

Here, `network` is the training network, and `"./MyNet.ckpt"` is the saving path of the network model.

### Saving the Model During Training

In the process of model training, use the `callbacks` parameter in `model.train` to pass in the object [ModelCheckpoint](https://mindspore.cn/docs/en/master/api_python/mindspore.train.html#mindsporetraincallbackmodelcheckpoint) that saves the model (Generally used with [CheckpointConfig](https://mindspore.cn/docs/en/master/api_python/mindspore.train.html#mindsporetraincallbackcheckpointconfig)), which can save the model parameters and generate CheckPoint (abbreviated as ckpt) files.

You can configure the checkpoint policies as required. The following describes the usage:

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# Set the number of epoch_num
epoch_num = 5

# Set the model saving parameters
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

# Apply the model saving parameters
ckpoint = ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)
model.train(epoch_num, dataset_train, callbacks=ckpoint)
```

In the preceding code, you need to initialize a `CheckpointConfig` class object to set the saving policy.

- `save_checkpoint_steps` indicates the interval (in steps) for saving the checkpoint file.
- `keep_checkpoint_max` indicates the maximum number of checkpoint files that can be retained.
- `prefix` indicates the prefix of the generated checkpoint file.
- `directory` indicates the directory for storing files.

Create a `ModelCheckpoint` object and pass it to the `model.train` method. Then the checkpoint function can be used during training.

The generated checkpoint file is as follows:

```text
lenet-graph.meta # Computational graph after compiled.
lenet-1_1875.ckpt  # The extension of the checkpoint file is .ckpt.
lenet-2_1875.ckpt  # The file name format contains the epoch and step correspond to the saved parameters. Here are the model parameters for the 1875th step of the 2nd epoch.
lenet-3_1875.ckpt  # indicates the model parameters saved for the 1875th step of the 3rd epoch.
...
```

If you use the same prefix and run the training script for multiple times, checkpoint files with the same name may be generated. To help users distinguish files generated each time, MindSpore adds underscores "_" and digits to the end of the user-defined prefix. If you want to delete the `.ckpt` file, delete the `.meta` file at the same time.

For example, `lenet_3-2_1875.ckpt` indicates the CheckPoint file of the 1875th step of the 2nd epoch generated by running the 3rd script.

## Loading the Model

To load the model weight, you need to create an instance of the same model and then use the `load_checkpoint` and `load_param_into_net` methods to load parameters.

The sample code is as follows:

```python
from mindspore import load_checkpoint, load_param_into_net

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet

# Store the model parameters in the parameter dictionary, where the model parameters saved during the training process above are loaded
param_dict = load_checkpoint("./lenet/lenet-1_1875.ckpt")

# Redefine a LeNet neural network
net = lenet(num_classes=10, pretrained=False)

# Load parameters to the network
load_param_into_net(network, param_dict)
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"accuracy"})
```

- The `load_checkpoint` method loads the network parameters in the parameter file to the `param_dict` dictionary.
- The `load_param_into_net` method loads the parameters in the `param_dict` dictionary to the network or optimizer. After the loading, parameters in the network are stored by the checkpoint.

### Validating the Model

After the above module loads the parameters into the network, for the inference scenario, you can call the `eval` function for inference verification. The sample code is as follows:

```python
# Call eval() for inference.
download_eval = Mnist(path="./mnist", split="test", batch_size=32, resize=32, download=True)
dataset_eval = download_eval.run()
acc = model.eval(dataset_eval)

print("{}".format(acc))
```

```text
{'accuracy': 0.9857772435897436}
```

### For Transfer Learning

For task interruption retraining and fine-tuning scenarios, you can call the `train` function for transfer learning. The sample code is as follows:

```python
# Define a training dataset.
download_train = Mnist(path="./mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

# Network model calls train() for training.
model.train(epoch_num, dataset_train, callbacks=[LossMonitor(0.01, 1875)])
```

```text
Epoch:[  0/  5], step:[ 1875/ 1875], loss:[0.000/0.010], time:2.193 ms, lr:0.01000
Epoch time: 4106.620 ms, per step time: 2.190 ms, avg loss: 0.010
Epoch:[  1/  5], step:[ 1875/ 1875], loss:[0.000/0.009], time:2.036 ms, lr:0.01000
Epoch time: 4233.697 ms, per step time: 2.258 ms, avg loss: 0.009
Epoch:[  2/  5], step:[ 1875/ 1875], loss:[0.000/0.010], time:2.045 ms, lr:0.01000
Epoch time: 4246.248 ms, per step time: 2.265 ms, avg loss: 0.010
Epoch:[  3/  5], step:[ 1875/ 1875], loss:[0.000/0.008], time:2.001 ms, lr:0.01000
Epoch time: 4235.036 ms, per step time: 2.259 ms, avg loss: 0.008
Epoch:[  4/  5], step:[ 1875/ 1875], loss:[0.002/0.008], time:2.039 ms, lr:0.01000
Epoch time: 4354.482 ms, per step time: 2.322 ms, avg loss: 0.008
```


