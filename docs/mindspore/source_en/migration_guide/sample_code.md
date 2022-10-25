# Network Migration Debugging Example

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/sample_code.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following uses the classic network ResNet-50 as an example to describe the network migration method in detail based on the code.

## Model Analysis and Preparation

Assume that the MindSpore operating environment has been configured according to [Environment Preparation and Information Acquisition](https://www.mindspore.cn/docs/en/master/migration_guide/enveriment_preparation.html). Assume that ResNet-50 has not been implemented in the models repository.

First, analyze the algorithm and network structure.

The Residual Neural Network (ResNet) was proposed by Kaiming He et al. from Microsoft Research Institute. They used residual units to successfully train a 152-layer neural network, and thus became the winner of ILSVRC 2015. A conventional convolutional network or fully-connected network has more or less information losses, and further causes gradient disappearance or explosion. As a result, deep network training fails. The ResNet can solve these problems to some extent. By passing the input information to the output, the information integrity is protected. The network only needs to learn the differences between the input and output, simplifying the learning objective and difficulty. Its structure can accelerate training of a neural network and greatly improve the accuracy of the network model.

[Paper](https://arxiv.org/pdf/1512.03385.pdf): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

The [sample code of PyTorch ResNet-50 CIFAR-10](https://gitee.com/mindspore/docs/tree/master/docs/mindspore/source_zh_cn/migration_guide/code/resnet_convert/resnet_pytorch) contains the PyTorch ResNet implementation, CIFAR-10 data processing, network training, and inference processes.

### Checklist

When reading the paper and referring to the implementation, analyze and fill in the following checklist:

|Trick|Record|
|----|----|
|Data augmentation| RandomCrop, RandomHorizontalFlip, Resize, Normalize|
|Learning rate attenuation policy| Fixed learning rate = 0.001|
|Optimization parameters| Adam optimizer, weight_decay = 1e-5|
|Training parameters| batch_size = 32, epochs = 90|
|Network structure optimization| Bottleneck |
|Training process optimization| None|

### Reproducing Reference Implementation

Download the PyTorch code and CIFAR-10 dataset to train the network.

```text
Train Epoch: 89 [0/1563 (0%)]    Loss: 0.010917
Train Epoch: 89 [100/1563 (6%)]    Loss: 0.013386
Train Epoch: 89 [200/1563 (13%)]    Loss: 0.078772
Train Epoch: 89 [300/1563 (19%)]    Loss: 0.031228
Train Epoch: 89 [400/1563 (26%)]    Loss: 0.073462
Train Epoch: 89 [500/1563 (32%)]    Loss: 0.098645
Train Epoch: 89 [600/1563 (38%)]    Loss: 0.112967
Train Epoch: 89 [700/1563 (45%)]    Loss: 0.137923
Train Epoch: 89 [800/1563 (51%)]    Loss: 0.143274
Train Epoch: 89 [900/1563 (58%)]    Loss: 0.088426
Train Epoch: 89 [1000/1563 (64%)]    Loss: 0.071185
Train Epoch: 89 [1100/1563 (70%)]    Loss: 0.094342
Train Epoch: 89 [1200/1563 (77%)]    Loss: 0.126669
Train Epoch: 89 [1300/1563 (83%)]    Loss: 0.245604
Train Epoch: 89 [1400/1563 (90%)]    Loss: 0.050761
Train Epoch: 89 [1500/1563 (96%)]    Loss: 0.080932

Test set: Average loss: -9.7052, Accuracy: 91%

Finished Training
```

You can download training logs and saved parameter files from [resnet_pytorch_res](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/resnet_pytorch_res.zip).

### Analyzing API/Feature Missing

- API analysis

| PyTorch API      | MindSpore API| Different or Not|
| ---------------------- | ------------------ | ------|
| `nn.Conv2D`            | `nn.Conv2d`        | Yes. [Difference](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/nn_Conv2d.html)|
| `nn.BatchNorm2D`       | `nn.BatchNom2d`    | Yes. [Difference](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/BatchNorm2d.html)|
| `nn.ReLU`              | `nn.ReLU`          | No|
| `nn.MaxPool2D`         | `nn.MaxPool2d`     | Yes. [Difference](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/MaxPool2d.html)|
| `nn.AdaptiveAvgPool2D` | `nn.AdaptiveAvgPool2D` |  No |
| `nn.Linear`            | `nn.Dense`         | Yes. [Difference](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/Dense.html)|
| `torch.flatten`        | `nn.Flatten`       | No|

By checking [PyTorch API Mapping] (https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html), we find that four APIs are different.

- Function analysis

| PyTorch Function         | MindSpore Function                   |
| ------------------------- | ------------------------------------- |
| `nn.init.kaiming_normal_` | `initializer(init='HeNormal')`        |
| `nn.init.constant_`       | `initializer(init='Constant')`        |
| `nn.Sequential`           | `nn.SequentialCell`                   |
| `nn.Module`               | `nn.Cell`                             |
| `nn.distibuted`           | `set_auto_parallel_context`   |
| `torch.optim.SGD`         | `nn.optim.SGD` or `nn.optim.Momentum` |

(The interface design of MindSpore is different from that of PyTorch. Therefore, only the comparison of key functions is listed here.)

After API and function analysis, we find that there are no missing APIs and functions on MindSpore compared with PyTorch.

## MindSpore Model Implementation

### Datasets

The CIFAR-10 dataset of PyTorch is processed as follows:

```python
import torch
import torchvision.transforms as trans
import torchvision

train_transform = trans.Compose([
    trans.RandomCrop(32, padding=4),
    trans.RandomHorizontalFlip(0.5),
    trans.Resize(224),
    trans.ToTensor(),
    trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
test_transform = trans.Compose([
    trans.Resize(224),
    trans.RandomHorizontalFlip(0.5),
    trans.ToTensor(),
    trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
```

If the CIFAR-10 dataset does not exist on the local host, you can add `download=True` when using `torchvision.datasets.CIFAR10` to automatically download the dataset.

The CIFAR-10 dataset directory is organized as follows:

```text
└─dataset_path
    ├─cifar-10-batches-bin      # train dataset
        ├─ data_batch_1.bin
        ├─ data_batch_2.bin
        ├─ data_batch_3.bin
        ├─ data_batch_4.bin
        ├─ data_batch_5.bin
    └─cifar-10-verify-bin       # evaluate dataset
        ├─ test_batch.bin
```

This operation is implemented on MindSpore as follows:

```python
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset.transforms.transforms import TypeCast

def create_cifar_dataset(dataset_path, do_train, batch_size=32, image_size=(224, 224), rank_size=1, rank_id=0):
    dataset = ds.Cifar10Dataset(dataset_path, shuffle=do_train,
                                num_shards=rank_size, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize(image_size),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    type_cast_op = TypeCast(ms.int32)

    data_set = dataset.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=trans, input_columns="image")

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=do_train)
    return data_set
```

### Network Model Implementation

By referring to [PyTorch ResNet](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/code/resnet_convert/resnet_pytorch/resnet.py), we have implemented [MindSpore ResNet](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/code/resnet_convert/resnet_ms/src/resnet.py). The comparison tool shows that the implementation is different in the following aspects:

```python
# Conv2d PyTorch
nn.Conv2d(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    groups=groups,
    bias=False,
    dilation=dilation,
)
##########################################

# Conv2d MindSpore
nn.Conv2d(
    in_planes,
    out_planes,
    kernel_size=3,
    pad_mode="pad",
    stride=stride,
    padding=dilation,
    group=groups,
    has_bias=False,
    dilation=dilation,
)
```

```python
# PyTorch
nn.Module
############################################
# MindSpore
nn.Cell
```

```python
# PyTorch
nn.ReLU(inplace=True)
############################################
# MindSpore
nn.ReLU()
```

```python
# PyTorch graph construction
forward
############################################
# MindSpore graph construction
construct
```

```python
# PyTorch MaxPool2d with padding
maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
############################################
# MindSpore MaxPool2d with padding
maxpool = nn.SequentialCell([
              nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
              nn.MaxPool2d(kernel_size=3, stride=2)])
```

```python
# PyTorch AdaptiveAvgPool2d
avgpool = nn.AdaptiveAvgPool2d((1, 1))
############################################
# When PyTorch AdaptiveAvgPool2d output shape is set to 1, MindSpore ReduceMean functions the same with higher speed.
mean = ops.ReduceMean(keep_dims=True)
```

```python
# PyTorch full connection
fc = nn.Linear(512 * block.expansion, num_classes)
############################################
# MindSpore full connection
fc = nn.Dense(512 * block.expansion, num_classes)
```

```python
# PyTorch Sequential
nn.Sequential
############################################
# MindSpore SequentialCell
nn.SequentialCell
```

```python
# PyTorch initialization
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Zero-initialize the last BN in each residual branch,
# so that the residual branch starts with zeros, and each residual block behaves like an identity.
# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
if zero_init_residual:
    for m in self.modules():
        if isinstance(m, Bottleneck) and m.bn3.weight is not None:
            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

############################################

# MindSpore initialization
for _, cell in self.cells_and_names():
    if isinstance(cell, nn.Conv2d):
        cell.weight.set_data(ms.common.initializer.initializer(
            ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
            cell.weight.shape, cell.weight.dtype))
    elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
        cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
        cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
    elif isinstance(cell, (nn.Dense)):
        cell.weight.set_data(ms.common.initializer.initializer(
            ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
            cell.weight.shape, cell.weight.dtype))
        cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

# Zero-initialize the last BN in each residual branch,
# so that the residual branch starts with zeros, and each residual block behaves like an identity.
# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
if zero_init_residual:
    for _, cell in self.cells_and_names():
        if isinstance(cell, Bottleneck) and cell.bn3.gamma is not None:
            cell.bn3.gamma.set_data("zeros", cell.bn3.gamma.shape, cell.bn3.gamma.dtype)
        elif isinstance(cell, BasicBlock) and cell.bn2.weight is not None:
            cell.bn2.gamma.set_data("zeros", cell.bn2.gamma.shape, cell.bn2.gamma.dtype)
```

### Loss Function

PyTorch:

```python
net_loss = torch.nn.CrossEntropyLoss()
```

MindSpore:

```python
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
```

### Learning Rate and Optimizer

PyTorch:

```python
net_opt = torch.optim.Adam(net.parameters(), 0.001, weight_decay=1e-5)
```

MindSpore:

```python
optimizer = nn.Adam(resnet.trainable_params(), 0.001, weight_decay=1e-5)
```

## Model Validation

The trained PyTorch parameters are obtained in [Reproducing Reference Implementation](#reproducing-reference-implementation). How do I convert the parameter file into a checkpoint file that can be used by MindSpore?

The following steps are required:

1. Print the names and shapes of all parameters in the PyTorch parameter file and the names and shapes of all parameters in the MindSpore cell to which parameters need to be loaded.
2. Compare the parameter name and shape to construct the parameter mapping.
3. Create a parameter list based on the parameter mapping (PyTorch parameters -> numpy -> MindSpore parameters) and save the parameter list as a checkpoint.
4. Unit test: Load PyTorch parameters and MindSpore parameters, construct random input, and compare the output.

### Printing Parameters

```python
import torch
# Print the parameter names and shapes of all parameters in the PyTorch parameter file and return the parameter dictionary.
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params

# Print the names and shapes of all parameters in the MindSpore cell and return the parameter dictionary.
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params
```

Run the following code:

```python
from resnet_ms.src.resnet import resnet50 as ms_resnet50
pth_path = "resnet.pth"
pt_param = pytorch_params(pth_path)
print("="*20)
ms_param = mindspore_params(ms_resnet50(num_classes=10))
```

You can obtain the following result:

```text
conv1.weight (64, 3, 7, 7)
bn1.weight (64,)
bn1.bias (64,)
bn1.running_mean (64,)
bn1.running_var (64,)
bn1.num_batches_tracked ()
layer1.0.conv1.weight (64, 64, 1, 1)
......
===========================================
conv1.weight (64, 3, 7, 7)
bn1.moving_mean (64,)
bn1.moving_variance (64,)
bn1.gamma (64,)
bn1.beta (64,)
layer1.0.conv1.weight (64, 64, 1, 1)
......
```

### Parameter Mapping and Checkpoint Saving

Except the BatchNorm parameter, the names and shapes of other parameters are correct. In this case, you can write a simple Python script for parameter mapping.

```python
import mindspore as ms
def param_convert(ms_params, pt_params, ckpt_path):
    # Parameter name mapping dictionary
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    for ms_param in ms_params.keys():
        # In the parameter list, only the parameters that contain bn and downsample.1 are the parameters of the BatchNorm operator.
        if "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # If the corresponding parameter is found and the shape is the same, add the parameter to the parameter list.
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # Other parameters
        else:
            # If the corresponding parameter is found and the shape is the same, add the parameter to the parameter list.
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # Save as MindSpore checkpoint.
    ms.save_checkpoint(new_params_list, ckpt_path)

ckpt_path = "resnet50.ckpt"
param_convert(ms_params, pt_params, ckpt_path)
```

After the execution is complete, you can find the generated checkpoint file in `ckpt_path`.

If the parameter mapping is complex and it is difficult to find the mapping based on the parameter name, you can write a parameter mapping dictionary, for example:

```python
param = {
    'bn1.bias': 'bn1.beta',
    'bn1.weight': 'bn1.gamma',
    'IN.weight': 'IN.gamma',
    'IN.bias': 'IN.beta',
    'BN.bias': 'BN.beta',
    'in.weight': 'in.gamma',
    'bn.weight': 'bn.gamma',
    'bn.bias': 'bn.beta',
    'bn2.weight': 'bn2.gamma',
    'bn2.bias': 'bn2.beta',
    'bn3.bias': 'bn3.beta',
    'bn3.weight': 'bn3.gamma',
    'BN.running_mean': 'BN.moving_mean',
    'BN.running_var': 'BN.moving_variance',
    'bn.running_mean': 'bn.moving_mean',
    'bn.running_var': 'bn.moving_variance',
    'bn1.running_mean': 'bn1.moving_mean',
    'bn1.running_var': 'bn1.moving_variance',
    'bn2.running_mean': 'bn2.moving_mean',
    'bn2.running_var': 'bn2.moving_variance',
    'bn3.running_mean': 'bn3.moving_mean',
    'bn3.running_var': 'bn3.moving_variance',
    'downsample.1.running_mean': 'downsample.1.moving_mean',
    'downsample.1.running_var': 'downsample.1.moving_variance',
    'downsample.0.weight': 'downsample.1.weight',
    'downsample.1.bias': 'downsample.1.beta',
    'downsample.1.weight': 'downsample.1.gamma'
}
```

Then, you can obtain the parameter file based on the `param_convert` process.

### Unit Test

After obtaining the corresponding parameter file, you need to perform a unit test on the entire model to ensure model consistency.

```python
import numpy as np
import torch
import mindspore as ms
from resnet_ms.src.resnet import resnet50 as ms_resnet50
from resnet_pytorch.resnet import resnet50 as pt_resnet50

def check_res(pth_path, ckpt_path):
    inp = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
    # When performing a unit test, you need to add a training or inference label to the cell.
    ms_resnet = ms_resnet50(num_classes=10).set_train(False)
    pt_resnet = pt_resnet50(num_classes=10).eval()
    pt_resnet.load_state_dict(torch.load(pth_path, map_location='cpu'))
    ms.load_checkpoint(ckpt_path, ms_resnet)
    print("========= pt_resnet conv1.weight ==========")
    print(pt_resnet.conv1.weight.detach().numpy().reshape((-1,))[:10])
    print("========= ms_resnet conv1.weight ==========")
    print(ms_resnet.conv1.weight.data.asnumpy().reshape((-1,))[:10])
    pt_res = pt_resnet(torch.from_numpy(inp))
    ms_res = ms_resnet(ms.Tensor(inp))
    print("========= pt_resnet res ==========")
    print(pt_res)
    print("========= ms_resnet res ==========")
    print(ms_res)
    print("diff", np.max(np.abs(pt_res.detach().numpy() - ms_res.asnumpy())))

pth_path = "resnet.pth"
ckpt_path = "resnet50.ckpt"
check_res(pth_path, ckpt_path)
```

During the unit test, you need to add training or inference labels to cells. PyTorch training uses `.train()` and inference uses `.eval()`, MindSpore training uses `.set_train()` and inference uses `.set_train(False)`.

```text
========= pt_resnet conv1.weight ==========
[ 1.091892e-40 -1.819391e-39  3.509566e-40 -8.281730e-40  1.207908e-39
 -3.576954e-41 -1.000796e-39  1.115791e-39 -1.077758e-39 -6.031427e-40]
========= ms_resnet conv1.weight ==========
[ 1.091892e-40 -1.819391e-39  3.509566e-40 -8.281730e-40  1.207908e-39
 -3.576954e-41 -1.000796e-39  1.115791e-39 -1.077758e-39 -6.031427e-40]
========= pt_resnet res ==========
tensor([[-15.1945,  -5.6529,   6.5738,   9.7807,  -2.4615,   3.0365,  -4.7216,
         -11.1005,   2.7121,  -9.3612],
        [-14.2412,  -5.9004,   5.6366,   9.7030,  -1.6322,   2.6926,  -3.7307,
         -10.7582,   1.4195,  -7.9930],
        [-13.4795,  -5.6582,   5.6432,   8.9152,  -1.5169,   2.6958,  -3.4469,
         -10.5300,   1.3318,  -8.1476],
        [-13.6448,  -5.4239,   5.8254,   9.3094,  -2.1969,   2.7042,  -4.1194,
         -10.4388,   1.9331,  -8.1746]], grad_fn=<AddmmBackward0>)
========= ms_resnet res ==========
[[-15.194535   -5.652934    6.5737996   9.780719   -2.4615316   3.0365033
   -4.7215843 -11.100524    2.7121294  -9.361177 ]
 [-14.24116    -5.9004383   5.6366115   9.702984   -1.6322318   2.69261
   -3.7307222 -10.758192    1.4194587  -7.992969 ]
 [-13.47945    -5.658216    5.6432185   8.915173   -1.5169426   2.6957715
   -3.446888  -10.529953    1.3317728  -8.147601 ]
 [-13.644804   -5.423854    5.825424    9.309403   -2.1969485   2.7042081
   -4.119426  -10.438771    1.9330862  -8.174606 ]]
diff 2.861023e-06
```

The final result is similar and basically meets the expectation. If the result difference is large, you need to compare the result layer by layer.

## Inference Process

PyTorch inference:

```python
import torch
import torchvision.transforms as trans
import torchvision
import torch.nn.functional as F
from resnet import resnet50

def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(data_loader.dataset)))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
test_transform = trans.Compose([
    trans.Resize(224),
    trans.RandomHorizontalFlip(0.5),
    trans.ToTensor(),
    trans.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
# 2. define forward network
net = resnet50(num_classes=10).cuda() if use_cuda else resnet50(num_classes=10)
net.load_state_dict(torch.load("./resnet.pth", map_location='cpu'))
test_epoch(net, device, test_loader)
```

```text
Test set: Average loss: -9.7075, Accuracy: 91%
```

MindSpore implements this process:

```python
import numpy as np
import mindspore as ms
from mindspore import nn
from src.dataset import create_dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.utils import init_env
from src.resnet import resnet50


def test_epoch(model, data_loader, loss_func):
    model.set_train(False)
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        output = model(data)
        test_loss += float(loss_func(output, target).asnumpy())
        pred = np.argmax(output.asnumpy(), axis=1)
        correct += (pred == target.asnumpy()).sum()
    dataset_size = data_loader.get_dataset_size()
    test_loss /= dataset_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / dataset_size))


@moxing_wrapper()
def test_net():
    init_env(config)
    eval_dataset = create_dataset(config.dataset_name, config.data_path, False, batch_size=1,
                                  image_size=(int(config.image_height), int(config.image_width)))
    resnet = resnet50(num_classes=config.class_num)
    ms.load_checkpoint(config.checkpoint_path, resnet)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    test_epoch(resnet, eval_dataset, loss)


if __name__ == '__main__':
    test_net()
```

Run the following command:

```shell
python test.py --data_path data/cifar10/ --checkpoint_path resnet.ckpt
```

You can obtain the following inference accuracy result:

```text
run standalone!
Test set: Average loss: 0.3240, Accuracy: 91%
```

The inference accuracy is the same.

## Training Process

For details about the PyTorch training process, see [PyToch ResNet-50 CIFAR-10 Sample Code](https://gitee.com/mindspore/docs/tree/master/docs/mindspore/source_zh_cn/migration_guide/code/resnet_convert/resnet_pytorch). The log file and trained path are stored in [resnet_pytorch_res](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/resnet_pytorch_res.zip).

The corresponding MindSpore code is as follows:

```python
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.profiler import Profiler
from src.dataset import create_dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.utils import init_env
from src.resnet import resnet50


def train_epoch(epoch, model, data_loader):
    model.set_train()
    dataset_size = data_loader.get_dataset_size()
    for batch_idx, (data, target) in enumerate(data_loader):
        loss = float(model(data, target)[0].asnumpy())
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, dataset_size,
                100. * batch_idx / dataset_size, loss))


def test_epoch(model, data_loader, loss_func):
    model.set_train(False)
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        output = model(data)
        test_loss += float(loss_func(output, target).asnumpy())
        pred = np.argmax(output.asnumpy(), axis=1)
        correct += (pred == target.asnumpy()).sum()
    dataset_size = data_loader.get_dataset_size()
    test_loss /= dataset_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / dataset_size))


@moxing_wrapper()
def train_net():
    init_env(config)
    if config.enable_profiling:
        profiler = Profiler()
    train_dataset = create_dataset(config.dataset_name, config.data_path, True, batch_size=config.batch_size,
                                   image_size=(int(config.image_height), int(config.image_width)),
                                   rank_size=40, rank_id=config.rank_id)
    eval_dataset = create_dataset(config.dataset_name, config.data_path, False, batch_size=1,
                                  image_size=(int(config.image_height), int(config.image_width)))
    config.steps_per_epoch = train_dataset.get_dataset_size()
    resnet = resnet50(num_classes=config.class_num)
    optimizer = nn.Adam(resnet.trainable_params(), config.lr, weight_decay=config.weight_decay)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_net = nn.TrainOneStepWithLossScaleCell(
        nn.WithLossCell(resnet, loss), optimizer, ms.Tensor(config.loss_scale, ms.float32))
    for epoch in range(config.epoch_size):
        train_epoch(epoch, train_net, train_dataset)
        test_epoch(resnet, eval_dataset, loss)

    print('Finished Training')
    save_path = './resnet.ckpt'
    ms.save_checkpoint(resnet, save_path)


if __name__ == '__main__':
    train_net()
```

## Performance Optimization

During the preceding training, it is found that the training is slow and performance optimization is required. Before performing specific optimization items, run the profiler tool to obtain the performance data. The profiler tool can obtain only the training encapsulated by the model. Therefore, you need to reconstruct the training process first.

```python
device_num = config.device_num
if config.use_profilor:
    profiler = Profiler()
    # Note that the profiling data should not be too large. Otherwise, the processing will be slow. In this example, if use_profilor is set to True, the original dataset is divided into 40 copies.
    device_num = 40
train_dataset = create_dataset(config.dataset_name, config.data_path, True, batch_size=config.batch_size,
                               image_size=(int(config.image_height), int(config.image_width)),
                               rank_size=device_num, rank_id=config.rank_id)
.....
loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
model = ms.Model(resnet, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale)
if config.use_profilor:
    # Note that the profiling data should not be too large. Otherwise, the processing will be slow.
    model.train(3, train_dataset, callbacks=[LossMonitor(), TimeMonitor()], dataset_sink_mode=True)
    profiler.analyse()
else:
    model.train(config.epoch_size, train_dataset, eval_dataset, callbacks=[LossMonitor(), TimeMonitor()],
                dataset_sink_mode=False)
```

Set `use_profilor=True`. The `data` directory is generated in the running directory. Rename the directory `profiler_v1` and run the `mindinsight start` command in the same directory.

![resnet_profiler1](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler1.png)

The following figure shows the MindInsight profiler page. (This analysis is performed in the Ascend environment, which is similar to that in the GPU. The CPU does not support profiler.) There are three parts on the page.

The first part is step trace, which is the most basic part for profiler. The data of a single device includes the step interval and forward and backward propagation. The forward and backward time is the actual running time of the model on the device, and the step interval time includes data processing, data printing, and time when parameters are saved on the CPU during the training process.
It can be seen that the step trace time and forward and backward execution time are almost even, and non-device operations such as data processing account for a large part.

The second part is the forward and backward network execution time, where you can view details.

![resnet_profiler2](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler2.png)

The upper part shows the proportion of each AI CORE operator to the total time, and the lower part shows the details of each operator.

![resnet_profiler3](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler3.png)

You can click an operator to obtain the execution time, scope, shape, and type of the operator.

In addition to the AI CORE operators, there may be AI CPU and HOST CPU operators on the network. These operators take more time than the AI CORE operators. You can click the tabs to view the time.

![resnet_profiler4](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler4.png)

In addition to viewing the operator performance, you can also view the raw data for analysis.

![resnet_profiler5](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler5.png)

Go to the `profiler_v1/profiler/` directory and click the `aicore_intermediate_0_type.csv` file to view the statistics of each operator. There are 30 AI Core operators in total. The total execution time is 37.526 ms.

![resnet_profiler6](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler6.png)

In addition, `aicore_intermediate_0_detail.csv` contains detailed data of each operator, which is similar to the operator details displayed in MindInsight. `ascend_timeline_display_0.json` is a timeline data file. For details, see [timeline](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html#timeline-analysis).

The third part is the performance data during data processing. You can view the data queue status in this part.

![resnet_profiler7](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler7.png)

And a queue status of each data processing operation:

![resnet_profiler8](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler8.png)

Now, let's analyze the process and solve the problem.

From the step trace, the step interval and forward and backward execution time are almost even. MindSpore provides an [on-device execution method](https://mindspore.cn/docs/zh-CN/master/design/overview.html) to concurrently process data and execute the network on the device. You only need to set `dataset_sink_mode=True` in `model.train`. Note that this configuration is `True` by default. When this configuration is enabled, one epoch returns the result of only one network. You are advised to change the value to `False` during debugging.

When `dataset_sink_mode=True` is set, the result of setting the profiler is as follows:

![resnet_profiler9](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler9.png)

The execution time is reduced by half.

Let's go on with the analysis and optimization. According to the execution time of forward and backward operators, `Cast` and `BatchNorm` account for almost 50%. Why are there so many `Cast`? According to [Constructing MindSpore Network](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/model_development.html), Conv, Sort, and TopK in the Ascend environment can only be float16. Therefore, the `Cast` operator is added before and after Conv calculation. The most direct method is to change the network calculation to float16. Only `Cast` is added before the network input and loss computation. The consumption of the `Cast` operator can be ignored. This involves the mixed precision policy of MindSpore.

MindSpore has three methods to use mixed precision:

1. Use `Cast` to convert the network input `cast` into `float16` and the loss input `cast` into `float32`.
2. Use the `to_float` method of `Cell`. For details, see [Network Entity and Loss Construction](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/model_and_loss.html).
3. Use the `amp_level` interface of the `Model` to perform mixed precision. For details, see [Automatic Mixed-Precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html#mixed-precision).

Use the third method to set `amp_level` in `Model` to `O3` and check the profiler result.

![resnet_profiler10](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler10.png)

Each step takes only 23 ms.

Finally, let's look at data processing.

![resnet_profiler11](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler11.png)

After the sink mode is added, there are two queues in total. The host queue is a queue in the memory. The dataset object continuously places the input data required by the network in the host queue.
The other is a data queue on the device. The data in the host queue is cached to the data queue, and the network directly obtains the model input from the data queue.

The host queue is empty in many places, indicating that the dataset is quickly taken away by the data queue when data is continuously generated. The data queue is almost full. Therefore, data can keep up with network training, and data processing is not the bottleneck of network training.

If most of the data queues are empty, you need to optimize the data performance. For example:

![resnet_profiler12](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/images/resnet_profiler12.png)

In the queue of each data processing operation, the last operator and the `batch` operator are empty for a long time. In this case, you can increase the degree of parallelism of the `batch` operator. For details, see [Data Processing Performance Tuning](https://www.mindspore.cn/tutorials/experts/en/master/dataset/optimize.html).

The code required for ResNet migration can be obtained from [code](https://gitee.com/mindspore/docs/tree/master/docs/mindspore/source_zh_cn/migration_guide/code).

You can click the following video to learn.

<div style="position: relative; padding: 30% 45%;">
<iframe style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="https://player.bilibili.com/player.html?aid=216889508&bvid=BV1sa411P737&cid=802191204&page=1&high_quality=1&&danmaku=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
</div>
