# Model Layers

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/layer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

MindSpore can use `Cell` as the base class to build the network structure.

To facilitate user operations, MindSpore provides a large number of built-in model layers, which can be directly called by using APIs.

You can also customize a model. For details, see "Building a Customized Network."

## Built-in Model Layers

The MindSpore framework provides abundant APIs at the layer of `mindspore.nn`. The APIs are as follows:

- Activation layer

  The activation layer has a large number of built-in activation functions, which are often used in defining the network structure. The activation function adds a nonlinear operation to the network, so that the network can have a better fitting effect.

  Main APIs include `Softmax`, `Relu`, `Elu`, `Tanh` and `Sigmoid`.

- Basic layer

  The basic layer implements some common basic structures on the network, such as the full connection layer, Onehot encoding, Dropout, and flat layer.

  Main APIs include `Dense`, `Flatten`, `Dropout`, `Norm` and `OneHot`.

- Container layer

  The main function of the container layer is to implement the data structures for storing multiple cells.

  Main APIs include `SequentialCell` and `CellList`.

- Convolutional layer

  Convolutional layer provides some convolution computation functions, such as common convolution, deep convolution, and convolution transposition.

  Main APIs include `Conv2d`, `Conv1d`, `Conv2dTranspose` and `Conv1dTranspose`.

- Pooling layer

  The pooling layer provides computation functions such as average pooling and maximum pooling.

  The main APIs are `AvgPool2d`, `MaxPool2d`, and `AvgPool1d`.

- Embedding layer

  The embedding layer provides the word embedding computation function to map input words into dense vectors.

  The main APIs include `Embedding`, `EmbeddingLookup` and `EmbeddingLookUpSplitMode`.

- Long short-term memory recurrent layer

  The long short-term memory recurrent layer provides the LSTM computation function. `LSTM` internally calls the `LSTMCell` API. The `LSTMCell` is an LSTM unit that performs operations on an LSTM layer. When operations at multiple LSTM network layers are involved, the `LSTM` API is used.

  The main APIs include `LSTM` and `LSTMCell`.

- Normalization layer

  The normalization layer provides some normalization methods, that is, converting data into a mean value and a standard deviation by means of linear transformation or the like.

  Main APIs include `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm` and `GlobalBatchNorm`.

- Mathematical computation layer

  The mathematical computation layer provides some computation functions formed by operators, for example, data generation and some other mathematical computations.

  Main APIs include `ReduceLogSumExp`, `Range`, `LinSpace` and `LGamma`.

- Image layer

  The image computation layer provides some functions related to matrix computing to transform and compute image data.

  Main APIs include `ImageGradients`, `SSIM`, `MSSSIM`, `PSNR` and `CentralCrop`.

- Quantization layer

  Quantization is to convert data from the float type to the int type within a data range. Therefore, the quantization layer provides some data quantization methods and model layer structure encapsulation.

  Main APIs include `Conv2dBnAct`, `DenseBnAct`, `Conv2dBnFoldQuant` and `LeakyReLUQuant`.

## Application Cases

Model layers of MindSpore are under `mindspore.nn`. The usage method is as follows:

```python
import mindspore.nn as nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out
```

The preceding network building case shows that the program calls the APIs of the `Conv2d`, `BatchNorm2d`, `ReLU`, `Flatten`, and `Dense` model layers.

It is defined in the `Net` initialization method and runs in the `construct` method. These model layer APIs are connected in sequence to form an executable network.
