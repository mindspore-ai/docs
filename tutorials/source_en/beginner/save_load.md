# Saving and Loading the Model

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/save_load.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The previous section describes how to adjust hyperparameters and train network models. During network model training, we want to save the intermediate and final results for fine-tuning and subsequent model deployment and inference. This section describes how to save and load a model.

## Saving the Model

We use the save_checkpoint provided by MindSpore to save the model, pass it in the network and the save path:

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal

# Define the neural network
class LeNet(nn.Cell):
    def __init__(self, num_classes=10, num_channel=1):
        super(LeNet, self).__init__()
        layers = [nn.Conv2d(num_channel, 6, 5, pad_mode='valid'),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(6, 16, 5, pad_mode='valid'),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Flatten(),
                  nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02)),
                  nn.ReLU(),
                  nn.Dense(120, 84, weight_init=Normal(0.02)),
                  nn.ReLU(),
                  nn.Dense(84, num_classes, weight_init=Normal(0.02))]
        # Network management with CellList
        self.build_block = nn.CellList(layers)

    def construct(self, x):
        # Execute network with a for loop
        for layer in self.build_block:
            x = layer(x)
        return x

# Instantiate training network
network = LeNet()

# Save the model parameters to the specified path
ms.save_checkpoint(network, "./MyNet.ckpt")
```

- The `save_checkpoint` method will save the network parameters to the specified path, where `network` is the training network and `". /MyNet.ckpt"` is the path where the network model is saved.

## Loading the Model

To load the model weight, you need to create an instance of the same model and then use the `load_checkpoint` and `load_param_into_net` methods to load parameters.

The sample code is as follows:

```python
import mindspore as ms

# Store the model parameters in the dictionary of parameter, and the model parameters saved during the training process above are loaded here
param_dict = ms.load_checkpoint("./MyNet.ckpt")

# Redefine a LeNet Neural Network
net = LeNet()

# Load parameters into the network and print the names of unloaded parameters
not_loaded_params = ms.load_param_into_net(net, param_dict)
print(not_loaded_params)
```

```text
[]
```

- The `load_checkpoint` method loads the network parameters in the parameter file to the `param_dict` dictionary.
- The `load_param_into_net` method loads the parameters in the `param_dict` dictionary to the network or optimizer. After the loading, parameters in the network are stored by the checkpoint.
