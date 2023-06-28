# Deep Probabilistic Programming

`Ascend` `GPU` `Whole Process` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/apply_deep_probability_programming.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

A deep learning model has a strong fitting capability, and the Bayesian theory has a good explainability. MindSpore Deep Probabilistic Programming (MDP) combines deep learning and Bayesian learning. By setting a network weight to distribution and introducing latent space distribution, MDP can sample the distribution and forward propagation, which introduces uncertainty and enhances the robustness and explainability of a model. MDP contains general-purpose and professional probabilistic learning programming languages. It is applicable to professional users as well as beginners because the probabilistic programming supports the logic for developing deep learning models. In addition, it provides a toolbox for deep probabilistic learning to expand Bayesian application functions.

The following describes applications of deep probabilistic programming in MindSpore. Before performing the practice, ensure that MindSpore 0.7.0-beta or a later version has been installed. The contents are as follows:

1. Describe how to use the [bnn_layers module](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/bnn_layers) to implement the Bayesian neural network (BNN).
2. Describe how to use the [variational module](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/infer/variational) and [dpn module](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/dpn) to implement the Variational Autoencoder (VAE).
3. Describe how to use the [transforms module](https://gitee.com/mindspore/mindspore/tree/r1.1/mindspore/nn/probability/transforms) to implement one-click conversion from deep neural network (DNN) to BNN.
4. Describe how to use the [toolbox module](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/nn/probability/toolbox/uncertainty_evaluation.py) to implement uncertainty evaluation.

## Using BNN

BNN is a basic model composed of probabilistic model and neural network. Its weight is not a definite value, but a distribution. The following example describes how to use the bnn_layers module in MDP to implement a BNN, and then use the BNN to implement a simple image classification function. The overall process is as follows:

1. Process the MNIST dataset.
2. Define the Bayes LeNet.
3. Define the loss function and optimizer.
4. Load and train the dataset.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/bnn_layers>.

### Processing the Dataset

The MNIST dataset is used in this example. The data processing is the same as that of [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html) in the tutorial.

### Defining the BNN

Bayesian LeNet is used in this example. The method of building a BNN by using the bnn_layers module is the same as that of building a common neural network. Note that `bnn_layers` and common neural network layers can be combined with each other.

```python
import mindspore.nn as nn
from mindspore.nn.probability import bnn_layers
import mindspore.ops as ops

class BNNLeNet5(nn.Cell):
    """
    bayesian Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> BNNLeNet5(num_class=10)

    """
    def __init__(self, num_class=10):
        super(BNNLeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = bnn_layers.ConvReparam(1, 6, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.conv2 = bnn_layers.ConvReparam(6, 16, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.fc1 = bnn_layers.DenseReparam(16 * 5 * 5, 120)
        self.fc2 = bnn_layers.DenseReparam(120, 84)
        self.fc3 = bnn_layers.DenseReparam(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

### Defining the Loss Function and Optimizer

A loss function and an optimizer need to be defined. The loss function is a training objective of the deep learning, and is also referred to as an objective function. The loss function indicates the distance between a logit of a neural network and a label, and is scalar data.

Common loss functions include mean square error, L2 loss, Hinge loss, and cross entropy. Cross entropy is usually used for image classification.

The optimizer is used for neural network solution (training). Because of the large scale of neural network parameters, the stochastic gradient descent (SGD) algorithm and its improved algorithm are used in deep learning to solve the problem. MindSpore encapsulates common optimizers, such as `SGD`, `Adam`, and `Momemtum`. In this example, the `Adam` optimizer is used. Generally, two parameters need to be set: learning rate (`learning_rate`) and weight attenuation (`weight_decay`).

An example of the code for defining the loss function and optimizer in MindSpore is as follows:

```python
# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
```

### Training the Network

The training process of BNN is similar to that of DNN. The only difference is that `WithLossCell` is replaced with `WithBNNLossCell` applicable to BNN. In addition to the `backbone` and `loss_fn` parameters, the `dnn_factor` and `bnn_factor` parameters are added to `WithBNNLossCell`. `dnn_factor` is a coefficient of the overall network loss computed by a loss function, and `bnn_factor` is a coefficient of the KL divergence of each Bayesian layer. The two parameters are used to balance the overall network loss and the KL divergence of the Bayesian layer, preventing the overall network loss from being covered by a large KL divergence.

```python
net_with_loss = bnn_layers.WithBNNLossCell(network, criterion, dnn_factor=60000, bnn_factor=0.000001)
train_bnn_network = TrainOneStepCell(net_with_loss, optimizer)
train_bnn_network.set_train()

train_set = create_dataset('./mnist_data/train', 64, 1)
test_set = create_dataset('./mnist_data/test', 64, 1)

epoch = 10

for i in range(epoch):
    train_loss, train_acc = train_model(train_bnn_network, network, train_set)

    valid_acc = validate_model(network, test_set)

    print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tvalidation Accuracy: {:.4f}'.
          format(i, train_loss, train_acc, valid_acc))
```

The code examples of `train_model` and `validate_model` in MindSpore are as follows:

```python
def train_model(train_net, net, dataset):
    accs = []
    loss_sum = 0
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = data['image']
        label = data['label']
        loss = train_net(train_x, label)
        output = net(train_x)
        log_output = ops.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)
        loss_sum += loss.asnumpy()

    loss_sum = loss_sum / len(accs)
    acc_mean = np.mean(accs)
    return loss_sum, acc_mean


def validate_model(net, dataset):
    accs = []
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = data['image']
        label = data['label']
        output = net(train_x)
        log_output = ops.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)

    acc_mean = np.mean(accs)
    return acc_mean
```

## Using the VAE

The following describes how to use the variational and dpn modules in MDP to implement VAE. VAE is a typical depth probabilistic model that applies variational inference to learn the representation of latent variables. The model can not only compress input data, but also generate new images of this type. The overall process is as follows:

1. Define a VAE.
2. Define the loss function and optimizer.
3. Process data.
4. Train the network.
5. Generate new samples or rebuild input samples.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/dpn>.

### Defining the VAE

Using the dpn module to build a VAE is simple. You only need to customize the encoder and decoder (DNN model) and call the `VAE` API.

```python
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


encoder = Encoder()
decoder = Decoder()
vae = VAE(encoder, decoder, hidden_size=400, latent_size=20)
```

### Defining the Loss Function and Optimizer

A loss function and an optimizer need to be defined. The loss function used in this example is `ELBO`, which is a loss function dedicated to variational inference. The optimizer used in this example is `Adam`.
An example of the code for defining the loss function and optimizer in MindSpore is as follows:

```python
# loss function definition
net_loss = ELBO(latent_prior='Normal', output_prior='Normal')

# optimization definition
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)

net_with_loss = nn.WithLossCell(vae, net_loss)
```

### Processing Data

The MNIST dataset is used in this example. The data processing is the same as that of [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html) in the tutorial.

### Training the Network

Use the `SVI` API in the variational module to train a VAE network.

```python
from mindspore.nn.probability.infer import SVI

vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
vae = vi.run(train_dataset=ds_train, epochs=10)
trained_loss = vi.get_train_loss()
```

You can use `vi.run` to obtain a trained network and use `vi.get_train_loss` to obtain the loss after training.

### Generating New Samples or Rebuilding Input Samples

With a trained VAE network, we can generate new samples or rebuild input samples.

```python
IMAGE_SHAPE = (-1, 1, 32, 32)
generated_sample = vae.generate_sample(64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'].asnumpy(), dtype=mstype.float32)
    reconstructed_sample = vae.reconstruct_sample(sample_x)
```

## One-click Conversion from DNN to BNN

For DNN researchers unfamiliar with the Bayesian model, MDP provides an advanced API `TransformToBNN` to convert the DNN model into the BNN model with one click. Currently, the API can be used in the LeNet, ResNet, MobileNet and VGG models. This example describes how to use the `TransformToBNN` API in the transforms module to convert DNNs into BNNs with one click. The overall process is as follows:

1. Define a DNN model.
2. Define the loss function and optimizer.
3. Function 1: Convert the entire model.
4. Function 2: Convert a layer of a specified type.

> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/transforms>.

### Defining the DNN Model

LeNet is used as a DNN model in this example.

```python
from mindspore.common.initializer import TruncatedNormal
import mindspore.nn as nn
import mindspore.ops as ops

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

The following shows the LeNet architecture.

```text
LeNet5
  (conv1) Conv2dinput_channels=1, output_channels=6, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (conv2) Conv2dinput_channels=6, output_channels=16, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (fc1) Densein_channels=400, out_channels=120, weight=Parameter (name=fc1.weight), has_bias=True, bias=Parameter (name=fc1.bias)
  (fc2) Densein_channels=120, out_channels=84, weight=Parameter (name=fc2.weight), has_bias=True, bias=Parameter (name=fc2.bias)
  (fc3) Densein_channels=84, out_channels=10, weight=Parameter (name=fc3.weight), has_bias=True, bias=Parameter (name=fc3.bias)
  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
```

### Defining the Loss Function and Optimizer

A loss function and an optimizer need to be defined. In this example, the cross entropy loss is used as the loss function, and `Adam` is used as the optimizer.

```python
network = LeNet5()

# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)

net_with_loss = WithLossCell(network, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)
```

### Instantiating TransformToBNN

The `__init__` function of `TransformToBNN` is defined as follows:

```python
class TransformToBNN:
    def __init__(self, trainable_dnn, dnn_factor=1, bnn_factor=1):
        net_with_loss = trainable_dnn.network
        self.optimizer = trainable_dnn.optimizer
        self.backbone = net_with_loss.backbone_network
        self.loss_fn = getattr(net_with_loss, "_loss_fn")
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.bnn_loss_file = None
```

The `trainable_bnn` parameter is a trainable DNN model packaged by `TrainOneStepCell`, `dnn_factor` and `bnn_factor` are the coefficient of the overall network loss computed by the loss function and the coefficient of the KL divergence of each Bayesian layer, respectively.
The code for instantiating `TransformToBNN` in MindSpore is as follows:

```python
from mindspore.nn.probability import transforms

bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
```

### Function 1: Converting the Entire Model

The `transform_to_bnn_model` method can convert the entire DNN model into a BNN model. The definition is as follows:

```python
    def transform_to_bnn_model(self,
                               get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias,
                                                          "out_channels": dp.out_channels, "activation": dp.activation},
                               get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels,
                                                         "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size,
                                                         "stride": dp.stride, "has_bias": dp.has_bias,
                                                         "padding": dp.padding, "dilation": dp.dilation,
                                                         "group": dp.group},
                               add_dense_args=None,
                               add_conv_args=None):
        r"""
        Transform the whole DNN model to BNN model, and wrap BNN model by TrainOneStepCell.

        Args:
            get_dense_args (function): The arguments gotten from the DNN full connection layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "has_bias": dp.has_bias}.
            get_conv_args (function): The arguments gotten from the DNN convolutional layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode,
                "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias}.
            add_dense_args (dict): The new arguments added to BNN full connection layer. Default: {}.
            add_conv_args (dict): The new arguments added to BNN convolutional layer. Default: {}.

        Returns:
            Cell, a trainable BNN model wrapped by TrainOneStepCell.
       """
```

The `get_dense_args` parameter specifies parameters to be obtained from the fully connected layer of a DNN model, and the `get_conv_args` parameter specifies parameters to be obtained from the convolutional layer of the DNN model, the `add_dense_args` and `add_conv_args` parameters specify new parameter values to be specified for the BNN layer. Note that parameters in `add_dense_args` cannot be the same as those in `get_dense_args`. The same rule applies to `add_conv_args` and `get_conv_args`.

The code for converting the entire DNN model into a BNN model in MindSpore is as follows:

```python
train_bnn_network = bnn_transformer.transform_to_bnn_model()
```

The structure of the converted model is as follows:

```text
LeNet5
  (conv1) ConvReparam
    in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, weight_mean=Parameter (name=conv1.weight_posterior.mean), weight_std=Parameter (name=conv1.weight_posterior.untransformed_std), has_bias=False
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (conv2) ConvReparam
    in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, weight_mean=Parameter (name=conv2.weight_posterior.mean), weight_std=Parameter (name=conv2.weight_posterior.untransformed_std), has_bias=False
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc1) DenseReparam
    in_channels=400, out_channels=120, weight_mean=Parameter (name=fc1.weight_posterior.mean), weight_std=Parameter (name=fc1.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc1.bias_posterior.mean), bias_std=Parameter (name=fc1.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc2) DenseReparam
    in_channels=120, out_channels=84, weight_mean=Parameter (name=fc2.weight_posterior.mean), weight_std=Parameter (name=fc2.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc2.bias_posterior.mean), bias_std=Parameter (name=fc2.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc3) DenseReparam
    in_channels=84, out_channels=10, weight_mean=Parameter (name=fc3.weight_posterior.mean), weight_std=Parameter (name=fc3.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc3.bias_posterior.mean), bias_std=Parameter (name=fc3.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
```

The convolutional layer and fully connected layer on the entire LeNet are converted into the corresponding Bayesian layers.

### Function 2: Converting a Layer of a Specified Type

The `transform_to_bnn_layer` method can convert a layer of a specified type (nn.Dense or nn.Conv2d) in the DNN model into a corresponding Bayesian layer. The definition is as follows:

```python
 def transform_to_bnn_layer(self, dnn_layer, bnn_layer, get_args=None, add_args=None):
        r"""
        Transform a specific type of layers in DNN model to corresponding BNN layer.

        Args:
            dnn_layer_type (Cell): The type of DNN layer to be transformed to BNN layer. The optional values are
            nn.Dense, nn.Conv2d.
            bnn_layer_type (Cell): The type of BNN layer to be transformed to. The optional values are
                DenseReparameterization, ConvReparameterization.
            get_args (dict): The arguments gotten from the DNN layer. Default: None.
            add_args (dict): The new arguments added to BNN layer. Default: None.

        Returns:
            Cell, a trainable model wrapped by TrainOneStepCell, whose sprcific type of layer is transformed to the corresponding bayesian layer.
        """
```

The `dnn_layer` parameter specifies a type of a DNN layer to be converted into a BNN layer, and the `bnn_layer` parameter specifies a type of a BNN layer to be converted into a DNN layer, `get_args` and `add_args` specify parameters obtained from the DNN layer and the parameters to be re-assigned to the BNN layer, respectively.

The code for converting a Dense layer in a DNN model into a corresponding Bayesian layer `DenseReparam` in MindSpore is as follows:

```python
train_bnn_network = bnn_transformer.transform_to_bnn_layer(nn.Dense, bnn_layers.DenseReparam)
```

The network structure after the conversion is as follows:

```text
LeNet5
  (conv1) Conv2dinput_channels=1, output_channels=6, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (conv2) Conv2dinput_channels=6, output_channels=16, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (fc1) DenseReparam
    in_channels=400, out_channels=120, weight_mean=Parameter (name=fc1.weight_posterior.mean), weight_std=Parameter (name=fc1.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc1.bias_posterior.mean), bias_std=Parameter (name=fc1.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc2) DenseReparam
    in_channels=120, out_channels=84, weight_mean=Parameter (name=fc2.weight_posterior.mean), weight_std=Parameter (name=fc2.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc2.bias_posterior.mean), bias_std=Parameter (name=fc2.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (fc3) DenseReparam
    in_channels=84, out_channels=10, weight_mean=Parameter (name=fc3.weight_posterior.mean), weight_std=Parameter (name=fc3.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc3.bias_posterior.mean), bias_std=Parameter (name=fc3.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None

    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1

    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None


  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
```

As shown in the preceding information, the convolutional layer on the LeNet remains unchanged, and the fully connected layer becomes the corresponding Bayesian layer `DenseReparam`.

## Using the Uncertainty Evaluation Toolbox

One of advantages of BNN is that uncertainty can be obtained. MDP provides a toolbox for uncertainty evaluation at the upper layer. Users can easily use the toolbox to compute uncertainty. Uncertainty means an uncertain degree of a prediction result of a deep learning model. Currently, most deep learning algorithm can only provide prediction results but cannot determine the result reliability. There are two types of uncertainties: aleatoric uncertainty and epistemic uncertainty.

- Aleatoric uncertainty: Internal noises in data, that is, unavoidable errors. This uncertainty cannot be reduced by adding sampling data.
- Epistemic uncertainty: An inaccurate evaluation of input data by a model due to reasons such as poor training or insufficient training data. This may be reduced by adding training data.

The uncertainty evaluation toolbox is applicable to mainstream deep learning models, such as regression and classification. During inference, developers can use the toolbox to obtain any aleatoric uncertainty and epistemic uncertainty by training models and training datasets and specifying tasks and samples to be evaluated. Developers can understand models and datasets based on uncertainty information.
> This example is for the GPU or Ascend 910 AI processor platform. You can download the complete sample code from <https://gitee.com/mindspore/mindspore/tree/r1.1/tests/st/probability/toolbox>.

The classification task is used as an example. The model is LeNet, the dataset is MNIST, and the data processing is the same as that of [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html) in the tutorial. To evaluate the uncertainty of the test example, use the toolbox as follows:

```python
from mindspore.nn.probability.toolbox.uncertainty_evaluation import UncertaintyEvaluation
from mindspore import load_checkpoint, load_param_into_net

network = LeNet5()
param_dict = load_checkpoint('checkpoint_lenet.ckpt')
load_param_into_net(network, param_dict)
# get train and eval dataset
ds_train = create_dataset('workspace/mnist/train')
ds_eval = create_dataset('workspace/mnist/test')
evaluation = UncertaintyEvaluation(model=network,
                                   train_dataset=ds_train,
                                   task_type='classification',
                                   num_classes=10,
                                   epochs=1,
                                   epi_uncer_model_path=None,
                                   ale_uncer_model_path=None,
                                   save_model=False)
for eval_data in ds_eval.create_dict_iterator():
    eval_data = Tensor(eval_data['image'].asnumpy(), mstype.float32)
    epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
    aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
```
