# Building a Customized Network

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/custom_net.md)

Both the network structure and the model layers (e.g. loss functions and optimizers mentioned above) are essentially a `Cell`. Therefore, they can be customized.

`Construct` a subclass inherited from `Cell`, define the operator and model layer in the `__init__` method, and build the network structure in the `construct` method.

Take the LeNet network as an example. Structure units such as the convolutional layer, pooling layer, and full connection layer are defined in the `__init__` method, and the defined content is connected together in the `construct` method to form a complete LeNet network structure.

The LeNet network is implemented as follows:

```python
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
