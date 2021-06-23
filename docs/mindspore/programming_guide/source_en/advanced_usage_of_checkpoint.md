# Saving, Loading and Converting Models

Translator: [franck23333](https://gitee.com/franck2333)

<!-- TOC -->

- [Saving, Loading and Converting Models](#saving-loading-and-converting-models)
    - [Summary](#summary)
    - [Preparation](#preparation)
    - [Advanced Usage](#advanced-usage)
        - [Saving](#saving)
            - [Saving CheckPoint Manually](#saving-checkpoint-manually)
            - [Saving the Specified Cell](#saving-the-specified-cell)
            - [Saving CheckPoint Asynchronously](#saving-checkpoint-asynchronously)
            - [Saving Custom Parameter Dictionary](#saving-custom-parameter-dictionary)
        - [Loading](#loading)
            - [Strictly Match the Parameter Name](#strictly-match-the-parameter-name)
            - [Filter Specified Prefix](#filter-specified-prefix)
        - [Converting CheckPoints of Other Frameworks to MindSpore Format](#converting-checkpoints-of-other-frameworks-to-mindspore-format)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/advanced_usage_of_checkpoint.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Summary

In the process of model training or model loading, sometimes it is necessary to replace some optimizers or other hyperparameters
in the model file and the fully-connected layer changes in the classification function, but you do not want
to change too much, or start training the model from the beginning. In view of this situation, MindSpore provides advanced usages
of CheckPoint that only adjusts parts of the weight of the model, and applies the method to the model tuning process.

For basic usage, please refer to [Saving Models](https://www.mindspore.cn/tutorial/training/en/master/use/save_model.html#checkpoint).

## Preparation

This article takes the LeNet network as an example to introduce the operation methods of saving, loading and converting
models in Mindspore.

Firstly, the following resources should be prepared:  

- MNIST dataset.

- The pretrained model file of the LeNet network: `checkpoint-lenet_1-1875.ckpt`.

- The data augmentation files `dataset_process.py`. For using the data augmentation method `create_dataset`, you can refer to the data agumentation method `create_dataset` defined in the official website [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/master/quick_start/quick_start.html).

- Define the LeNet network.  

Execute the following codes to complete the first three preparations.

```bash
mkdir -p ./datasets/MNIST_Data/train ./datasets/MNIST_Data/test
wget -NP ./datasets/MNIST_Data/train https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte --no-check-certificate
wget -NP ./datasets/MNIST_Data/train https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte --no-check-certificate
wget -NP ./datasets/MNIST_Data/test https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte --no-check-certificate
wget -NP ./datasets/MNIST_Data/test https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte --no-check-certificate
wget https://mindspore-website.obs.myhuaweicloud.com/notebook/source-codes/dataset_process.py -N --no-check-certificate
wget -N https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/checkpoint_lenet-1_1875.zip --no-check-certificate
unzip -o checkpoint_lenet-1_1875.zip
```

Define the LeNet network, the specific definition process is as follows.  

```python
from mindspore.common.initializer import Normal
import mindspore.nn as nn

class LeNet5(nn.Cell):
    """Lenet network structure."""
    def __init__(self, num_class=10, num_channel=1):
        """define the operator required"""
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16*5*5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """use the preceding operators to construct networks"""
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Advanced Usage

### Saving

#### Saving CheckPoint Manually

Use `save_checkpoint` to save CheckPoint files manually.

Application Scenarios:

1. Saving the initial values of the network.

2. Saving the specified network manually.

Execute the following code, after training 100 batches of the dataset for the pretrained model `checkpoint_lenet-1_1875.ckpt`,
use `save_checkpoint` to save the model `mindspore_lenet.ckpt` manually.  

```python
from mindspore import Model, load_checkpoint, save_checkpoint, load_param_into_net
from mindspore import context, Tensor
from dataset_process import create_dataset
import mindspore.nn as nn

network = LeNet5()
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
net_loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

params = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
load_param_into_net(network, params)

net_with_criterion = nn.WithLossCell(network, net_loss)
train_net = nn.TrainOneStepCell(net_with_criterion, net_opt)
train_net.set_train()

train_path = "./datasets/MNIST_Data/train"
ds_train = create_dataset(train_path)

count = 0
for item in ds_train.create_dict_iterator():
    input_data = item["image"]
    labels = item["label"]
    train_net(input_data, labels)
    count += 1
    if count==100:
        print(train_net.trainable_params())
        save_checkpoint(train_net, "mindspore_lenet.ckpt")
        break
```

```text
[Parameter (name=conv1.weight), Parameter (name=conv2.weight), Parameter (name=fc1.weight), Parameter (name=fc1.bias),
Parameter (name=fc2.weight), Parameter (name=fc2.bias), Parameter (name=fc3.weight), Parameter (name=fc3.bias),
Parameter (name=learning_rate), Parameter (name=momentum), Parameter (name=moments.conv1.weight),
Parameter (name=moments.conv2.weight), Parameter (name=moments.fc1.weight), Parameter (name=moments.fc1.bias),
Parameter (name=moments.fc2.weight), Parameter (name=moments.fc2.bias), Parameter (name=moments.fc3.weight),
Parameter (name=moments.fc3.bias)]
```

The weight parameters of `mindspore_lenet.ckpt` can be seen from the above printed information, including the weight
parameters of each hidden layer, learning rate, optimization rate in the process of forward propagation and the weight of optimizer function
in back propagation.

#### Saving the Specified Cell

Usage: The parameter `saved_network` of the Class `CheckpointConfig`.

Application Scenarios:  

- Only save the parameters of the inference network model (Not saving the optimizer's parameters halves the size of the generated CheckPoint file)

- Save subnet parameters to be used to the Fine-tune tasks.  

Use the method `CheckpointConfig` in the callback function and specify the cell of the saved model as `network`, that is, the forward propagation LeNet network.  

```python
import os
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

ds_train = create_dataset(train_path)
epoch_size = 1
model = Model(train_net)
config_ck = CheckpointConfig(saved_network=network)
ckpoint = ModelCheckpoint(prefix="lenet", config=config_ck)
model.train(epoch_size, ds_train, callbacks=[ckpoint, LossMonitor(625)])
```

```text
epoch: 1 step: 625, loss is 0.116291314  
epoch: 1 step: 1250, loss is 0.09527888  
epoch: 1 step: 1875, loss is 0.23090823
```

The model is saved to the model file`lenet-1_1875.ckpt` after training. Next, compare the size and specific weight between the cell of the specified saved model and the original model.  

```python
model_with_opt = os.path.getsize("./checkpoint_lenet-1_1875.ckpt") // 1024
params_without_change = load_checkpoint("./checkpoint_lenet-1_1875.ckpt")
print("with_opt size:", model_with_opt, "kB")
print(params_without_change)

print("\n=========after train===========\n")
model_without_opt = os.path.getsize("./lenet-1_1875.ckpt") // 1024
params_with_change = load_checkpoint("./lenet-1_1875.ckpt")
print("without_opt size:", model_without_opt, "kB")
print(params_with_change)
```

```text
with_opt size: 482 kB  
{'conv1.weight': Parameter (name=conv1.weight), 'conv2.weight': Parameter (name=conv2.weight),
'fc1.weight': Parameter (name=fc1.weight), 'fc1.bias': Parameter (name=fc1.bias),
'fc2.weight': Parameter (name=fc2.weight), 'fc2.bias': Parameter (name=fc2.bias),
'fc3.weight': Parameter (name=fc3.weight), 'fc3.bias': Parameter (name=fc3.bias),
'learning_rate': Parameter (name=learning_rate), 'momentum': Parameter (name=momentum),
'moments.conv1.weight': Parameter (name=moments.conv1.weight), 'moments.conv2.weight': Parameter (name=moments.conv2.weight),
'moments.fc1.weight': Parameter (name=moments.fc1.weight), 'moments.fc1.bias': Parameter (name=moments.fc1.bias),
'moments.fc2.weight': Parameter (name=moments.fc2.weight), 'moments.fc2.bias': Parameter (name=moments.fc2.bias),
'moments.fc3.weight': Parameter (name=moments.fc3.weight), 'moments.fc3.bias': Parameter (name=moments.fc3.bias)}  
=========after train===========  
without_opt size: 241 kB  
{'conv1.weight': Parameter (name=conv1.weight), 'conv2.weight': Parameter (name=conv2.weight),
'fc1.weight': Parameter (name=fc1.weight), 'fc1.bias': Parameter (name=fc1.bias), 'fc2.weight': Parameter (name=fc2.weight),
'fc2.bias': Parameter (name=fc2.bias), 'fc3.weight': Parameter (name=fc3.weight), 'fc3.bias': Parameter (name=fc3.bias)}
```

The weight file size of the saved model`lenet-1_1875.ckpt` is 241kB after training. Compared with the size 482kB of the original full model, the overall reduction is nearly half;

The specific comparison of the parameters in the model shows that the learning rate, optimization rate and the weight parameters related to reverse optimization are reduced in `lenet-1_1875.ckpt` compared with the parameters in `checkpoint_lenet-1_1875.ckpt`, and only the weight parameters of the forward propagation network LeNet are retained. It meets the expected result.

#### Saving CheckPoint Asynchronously

Usage: The parameters`async_save` of the Class`CheckpointConfig`.

Application Scenarios: The trained model has a large amount of parameters, and it can be saved during training, including the writing time that saves the CheckPoint file.  

```python
config_ck = CheckpointConfig(async_save=True)
ckpoint = ModelCheckpoint(prefix="lenet", config=config_ck)
model.train(epoch_size, ds_train, callbacks=ckpoint)
```

#### Saving Custom Parameter Dictionary

Usage: Construct an object `obj_dict` and pass it to the method `save_checkpoint`.

Application Scenarios:  

- During the training process, additional parameters (`lr`, `epoch_size`, etc.) need to be saved to CheckPoint files.

- Resave after modifying the parameters values in CheckPoint.

- Convert the CheckPoint files of PyTorch and TensorFlow into the CheckPoint files of MindSpore.  

There are two situations according to specific scenarios:  

1. When the CheckPoint file exists, resave it after modifying the content.  

    ```python
    params = load_checkpoint("./lenet-1_1875.ckpt")

    # eg: param_list = [{"name": param_name, "data": param_data},...]
    param_list = [{"name": k, "data":v} for k,v in params.items()]
    print("==========param_list===========\n")
    print(param_list)

    # del element
    del param_list[2]
    print("\n==========after delete param_list[2]===========\n")
    print(param_list)


    # add element "epoch_size"
    param = {"name": "epoch_size"}
    param["data"] = Tensor(10)
    param_list.append(param)
    print("\n==========after add element===========\n")
    print(param_list)

    # modify element
    param_list[3]["data"] = Tensor(66)
    # save a new checkpoint file
    print("\n==========after modify element===========\n")
    print(param_list)

    save_checkpoint(param_list, 'modify.ckpt')
    ```

    ```text
    ==========param_list===========  
    [{'name': 'conv1.weight', 'data': Parameter (name=conv1.weight)}, {'name': 'conv2.weight', 'data': Parameter (name=conv2.weight)},
    {'name': 'fc1.weight', 'data': Parameter (name=fc1.weight)}, {'name': 'fc1.bias', 'data': Parameter (name=fc1.bias)},
    {'name': 'fc2.weight', 'data': Parameter (name=fc2.weight)}, {'name': 'fc2.bias', 'data': Parameter (name=fc2.bias)},
    {'name': 'fc3.weight', 'data': Parameter (name=fc3.weight)}, {'name': 'fc3.bias', 'data': Parameter (name=fc3.bias)}]  
    ==========after delete param_list[2]===========  
    [{'name': 'conv1.weight', 'data': Parameter (name=conv1.weight)}, {'name': 'conv2.weight', 'data': Parameter (name=conv2.weight)},
    {'name': 'fc1.bias', 'data': Parameter (name=fc1.bias)}, {'name': 'fc2.weight', 'data': Parameter (name=fc2.weight)},
    {'name': 'fc2.bias', 'data': Parameter (name=fc2.bias)}, {'name': 'fc3.weight', 'data': Parameter (name=fc3.weight)},
    {'name': 'fc3.bias', 'data': Parameter (name=fc3.bias)}]  
    ==========after add element===========  
    [{'name': 'conv1.weight', 'data': Parameter (name=conv1.weight)}, {'name': 'conv2.weight', 'data': Parameter (name=conv2.weight)},
    {'name': 'fc1.bias', 'data': Parameter (name=fc1.bias)}, {'name': 'fc2.weight', 'data': Parameter (name=fc2.weight)},
    {'name': 'fc2.bias', 'data': Parameter (name=fc2.bias)}, {'name': 'fc3.weight', 'data': Parameter (name=fc3.weight)},
    {'name': 'fc3.bias', 'data': Parameter (name=fc3.bias)}, {'name': 'epoch_size', 'data': Tensor(shape=[], dtype=Int64, value= 10)}]  
    ==========after modify element===========  
    [{'name': 'conv1.weight', 'data': Parameter (name=conv1.weight)}, {'name': 'conv2.weight', 'data': Parameter (name=conv2.weight)},
    {'name': 'fc1.bias', 'data': Parameter (name=fc1.bias)}, {'name': 'fc2.weight', 'data': Tensor(shape=[], dtype=Int64, value= 66)},
    {'name': 'fc2.bias', 'data': Parameter (name=fc2.bias)}, {'name': 'fc3.weight', 'data': Parameter (name=fc3.weight)},
    {'name': 'fc3.bias', 'data': Parameter (name=fc3.bias)}, {'name': 'epoch_size', 'data': Tensor(shape=[], dtype=Int64, value= 10)}]  
    ```

    After the loaded model file is converted to the list type, the model parameters can be deleted, added, modified, etc.,
    and manually saved with `save_checkpoint` to complete the modification of the content of the model weight file.

2. Save the custom parameters list as a CheckPoint file.  

    ```python
    param_list = []
    # save epoch_size
    param = {"name": "epoch_size"}
    param["data"] = Tensor(10)
    param_list.append(param)

    # save learning rate
    param = {"name": "learning_rate"}
    param["data"] = Tensor(0.01)
    param_list.append(param)
    # save a new checkpoint file
    print(param_list)

    save_checkpoint(param_list, 'hyperparameters.ckpt')
    ```

    ```text
    [{'name': 'epoch_size', 'data': Tensor(shape=[], dtype=Int64, value= 10)}, {'name': 'learning_rate',
    'data': Tensor(shape=[], dtype=Float64, value= 0.01)}]
    ```

### Loading

#### Strictly Match the Parameter Name

When the weight parameters in the CheckPoint file are loaded into the `net`, the parameters with the same name in `net` and CheckPoint will be matched firstly.

After the matching is completed, if there are some parameters that are not loaded in the net, it will match the parameters with the same suffix name in ckpt and net.  

For example: The parameter value named `conv.0.weight` in CheckPoint will be loaded into the parameter named `net.conv.0.weight` in net.

If you want to cancel this kind of fuzzy matching, only adopt the strict matching mechanism, you can set it by the parameter `strict_load` in the method `load_param_into_net`.

The default is False, which means the fuzzy matching mechanism is adopted.  

```python
net = LeNet5()
params = load_checkpoint("lenet-1_1875.ckpt")
load_param_into_net(net, params, strict_load=True)
print("==========strict load mode===========")
print(params)
```

```text
==========strict load mode===========  
{'conv1.weight': Parameter (name=conv1.weight), 'conv2.weight': Parameter (name=conv2.weight), 'fc1.weight': Parameter (name=fc1.weight),
'fc1.bias': Parameter (name=fc1.bias), 'fc2.weight': Parameter (name=fc2.weight), 'fc2.bias': Parameter (name=fc2.bias),
'fc3.weight': Parameter (name=fc3.weight), 'fc3.bias': Parameter (name=fc3.bias)}  
```

#### Filter Specified Prefix

Usage: The parameter `filter_prefix` of `load_checkpoint`.

Application Scenarios: When loading the CheckPoint, you want to filter certain parameters that contain the specific prefix.

- When loading the CheckPoint, do not load the parameters in the optimizer (for example：filter_prefix=’moments’).

- Do not load the parameters of the convolutional layers (for example：filter_prefix=’conv1’).

```python
net = LeNet5()
print("=============net params=============")
params = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
load_param_into_net(net, params)
print(params)

net = LeNet5()
print("\n=============after filter_prefix moments=============")
params = load_checkpoint("checkpoint_lenet-1_1875.ckpt", filter_prefix='moments')
load_param_into_net(net, params)
print(params)
```

```text
=============net params=============  
{'conv1.weight': Parameter (name=conv1.weight), 'conv2.weight': Parameter (name=conv2.weight), 'fc1.weight': Parameter (name=fc1.weight),
'fc1.bias': Parameter (name=fc1.bias), 'fc2.weight': Parameter (name=fc2.weight), 'fc2.bias': Parameter (name=fc2.bias),
'fc3.weight': Parameter (name=fc3.weight), 'fc3.bias': Parameter (name=fc3.bias), 'learning_rate': Parameter (name=learning_rate),
'momentum': Parameter (name=momentum), 'moments.conv1.weight': Parameter (name=moments.conv1.weight), 'moments.conv2.weight':
Parameter (name=moments.conv2.weight), 'moments.fc1.weight': Parameter (name=moments.fc1.weight), 'moments.fc1.bias':
Parameter (name=moments.fc1.bias), 'moments.fc2.weight': Parameter (name=moments.fc2.weight), 'moments.fc2.bias':
Parameter (name=moments.fc2.bias), 'moments.fc3.weight': Parameter (name=moments.fc3.weight), 'moments.fc3.bias': Parameter (name=moments.fc3.bias)}  
=============after filter_prefix moments=============  
{'conv1.weight': Parameter (name=conv1.weight), 'conv2.weight': Parameter (name=conv2.weight), 'fc1.weight': Parameter (name=fc1.weight),
'fc1.bias': Parameter (name=fc1.bias), 'fc2.weight': Parameter (name=fc2.weight), 'fc2.bias': Parameter (name=fc2.bias),
'fc3.weight': Parameter (name=fc3.weight), 'fc3.bias': Parameter (name=fc3.bias), 'learning_rate': Parameter (name=learning_rate), 'momentum': Parameter (name=momentum)}
```

Using the mechanism of filtering prefixes, you can filter out the parameters that you do not want to load (in this case, it is the optimizer weight parameter). When performing Fine-tune, you can use other optimizers to optimize.

> The above is the advanced usage of MindSpore checkpoint, all the above usages can be used together.

### Converting CheckPoints of Other Frameworks to MindSpore Format

Convert CheckPoint files of other frameworks to the MindSpore format.

In general, the parameters names and parameters values are saved in the CheckPoint file. After invoking the loading interface
of the corresponding framework and obtaining the parameter names and values, construct the object according to the MindSpore format,
and then you can directly invoke the MindSpore interface to save as CheckPoint files in the MindSpore format.

The main work is to compare the parameter names between different frameworks, so that all parameter names in the network of the two frameworks correspond to each other (a map can be used for mapping).
The logic of the following code is transforming the parameter format, excluding the corresponding parameter name.

```python
import torch
from mindspore import Tensor, save_checkpoint

def pytorch2mindspore(default_file = 'torch_resnet.pth'):
    """read pth file"""
    par_dict = torch.load(default_file)['state_dict']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        params_list.append(param_dict)
    save_checkpoint(params_list,  'ms_resnet.ckpt')
```
