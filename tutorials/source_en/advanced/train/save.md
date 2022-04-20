# Saving and Exporting Models

`Ascend` `GPU` `CPU` `Model Export`

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/source_en/advanced/train/save.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## Overview

During model training, you can add CheckPoints to save model parameters for inference and retraining after interruption. If you want to perform inference on different hardware platforms, you need to generate corresponding MindIR, AIR and ONNX format files based on the network and CheckPoint format files.

- **CheckPoint**: The Protocol Buffers mechanism is adopted, which stores all the parameter values in the network. It is generally used to resume training after a training task is interrupted, or in a Fine Tune task after training.
- **MindIR**: MindSpore IR, is a kind of functional IR based on graph representation of MindSpore, which defines the extensible graph structure and the IR representation of the operator, and stores the network structure and weight parameter values. It eliminates model differences between different backends and is generally used to perform inference tasks across hardware platforms, such as performing inference on the Ascend 910 trained model on the Ascend 310, GPU, and MindSpore Lite side.
- **AIR**: Ascend Intermediate Representation, is an open file format defined by Huawei for machine learning, and stores network structure and weight parameter values, which can better adapt to Ascend AI processors. It is generally used to perform inference tasks on Ascend 310.
- **ONNX**: Open Neural Network Exchange, is an open file format designed for machine learning, storing both network structure and weight parameter values. Typically used for model migration between different frameworks or for use on the Inference Engine (TensorRT).

The following uses examples to describe how to save MindSpore CheckPoint files, and how to export MindIR, AIR and ONNX files.

## Saving the models

The [Save and Load section](https://mindspore.cn/tutorials/zh-CN/r1.7/beginner/save_load.html) of the beginner tutorials describes how to save model parameters directly using `save_checkpoint` and using the Callback mechanism to save model parameters during training. This section further describes how to save model parameters during training and use `save_checkpoint` save model parameters directly.

### Saving the models during the training

Saving model parameters during training. MindSpore provides two saving strategies, an iteration policy and a time policy, which can be set by creating a `CheckpointConfig` object. The iteration policy and the time policy cannot be used at the same time, where the iterative policy takes precedence over the time policy, and when set at the same time, only iteration policy can take effect. When the parameter display is set to None, the policy is abandoned. In addition, when an exception occurs during training, MindSpore also provides a breakpoint retrain function, that is, the system will automatically save the CheckPoint file when the exception occurs.

1. Iteration policy

    `CheckpointConfig` can be configured according to the number of iterations, and the parameters of the iteration policy are as follows:

    - `save_checkpoint_steps`: indicates how many CheckPoint files are saved every step, with a default value of 1.
    - `keep_checkpoint_max`: indicates how many CheckPoint files to save at most, with a default value of 5.

    ```python
    from mindspore.train.callback import CheckpointConfig

    # Save one CheckPoint file every 32 steps, and up to 10 CheckPoint files
    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
    ```

    In the case that the iteration policy script ends normally, the CheckPoint file of the last step is saved by default.

2. Time policy

    `CheckpointConfig` can be configured according to the training duration, and the parameters of the configuration time policy are as follows:

    - `save_checkpoint_seconds`: indicates how many seconds to save a CheckPoint file, with a default value of 0.
    - `keep_checkpoint_per_n_minutes`: indicates how many checkPoint files are kept every few minutes, with a default value of 0.

    ```python
    from mindspore.train.callback import CheckpointConfig

    # Save a CheckPoint file every 30 seconds and a CheckPoint file every 3 minutes
    config_ck = CheckpointConfig(save_checkpoint_seconds=30, keep_checkpoint_per_n_minutes=3)
    ```

    `save_checkpoint_seconds` parameters cannot be used with `save_checkpoint_steps` parameters. If both parameters are set, the `save_checkpoint_seconds` parameters are invalid.

3. Breakpoint renewal

    MindSpore provides a breakpoint renewal function, when the user turns on the function, if an exception occurs during training, MindSpore will automatically save the CheckPoint file (end-of-life CheckPoint) when the exception occurred. The function of breakpoint renewal is controlled by the `exception_save` parameter (bool type) in CheckpointConfig, which is turned on when set to True, and closed by False, which defaults to False. The end-of-life CheckPoint file saved by the breakpoint continuation function does not affect the CheckPoint saved in the normal process, and the naming mechanism and save path are consistent with the normal process settings, the only difference is that the '_breakpoint' will be added at the end of the end of the CheckPoint file name to distinguish. Its usage is as follows:

    ```python
    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

    # Configure the breakpoint continuation function to turn on
    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10, exception_save=True)
    ```

    If an exception occurs during training, the end-of-life CheckPoint is automatically saved, and if an exception occurs in the 10th step of the 10th epoch in the training, the saved end-of-life CheckPoint file is as follows.

    ```python
    resnet50-10_10_breakpoint.ckpt  # The end-of-life CheckPoint file name will be marked by '_breakpoint' to distinguish it from the normal process checkPoint.
    ```

### save_checkpoint saving models

You can use `save_checkpoint` function to save network weights to a CheckPoint file, and the common parameters are as follows:

- `save_obj`: Cell object or data list.
- `ckpt_file_name`: Checkpoint file name. If the file already exists, the original file will be overwritten.
- `integrated_save`: Whether to merge or save split Tensor in parallel scenarios. The default value is True.
- `async_save`: Whether to execute asynchronously save the checkpoint file. The default value is False.
- `append_dict`: Additional information that needs to be saved. The key of dict must be of type str, and the value type of dict must be float or bool. The default value is None.

1. `save_obj` parameter

    The [Save and Load section](https://mindspore.cn/tutorials/zh-CN/r1.7/beginner/save_load.html) of the beginner tutorials describes how to save model parameters directly using `save_checkpoint` when `save_obj` is a Cell object. Here's how to save model parameters when you pass in a list of data. When passing in a data list, each element of the list is of dictionary type, such as [{"name": param_name, "data": param_data} ,...], `param_name` type must be str, and the type of `param_data` must be Parameter or Tensor. An example is shown below:

    ```python
    from mindspore import save_checkpoint, Tensor
    from mindspore import dtype as mstype

    save_list = [{"name": "lr", "data": Tensor(0.01, mstype.float32)}, {"name": "train_epoch", "data": Tensor(20, mstype.int32)}]
    save_checkpoint(save_list, "hyper_param.ckpt")
    ```

2. `integrated_save` parameter

    indicates whether the parameters are saved in a merge, and the default is True. In the model parallel scenario, Tensor is split into programs run by different cards. If integrated_save is set to True, these split Tensors are merged and saved in each checkpoint file, so that the checkpoint file saves the complete training parameters.

    ```python
    save_checkpoint(net, "resnet50-2_32.ckpt", integrated_save=True)
    ```

3. `async_save` parameter

    indicates whether the asynchronous save function is enabled, which defaults to False. If set to True, multithreading is turned on to write checkpoint files, allowing training and save tasks to be performed in parallel, saving the total time the script runs when training large-scale networks.

    ```python
    save_checkpoint(net, "resnet50-2_32.ckpt", async_save=True)
    ```

4. `append_dict` parameter

    additional information needs to be saved, the type is dict type, and currently only supports the preservation of basic types, including int, float, bool, etc

    ```python
    save_dict = {"epoch_num": 2, "lr": 0.01}
    # In addition to the parameters in net, the information save_dict is also saved in the ckpt file
    save_checkpoint(net, "resnet50-2_32.ckpt",append_dict=save_dict)
    ```

## Transfer Learning

In the transfer learning scenario, when using a pre-trained model for training, the model parameters in the CheckPoint file cannot be used directly, and they need to be modified according to the actual situation to be suitable for the current network model. This section describes how to remove the fully connected layer parameter from a pre-trained model for Resnet50.

First download the [pre-trained model of Resnet50](https://download.mindspore.cn/vision/classification/resnet50_224.ckpt), which is trained on the ImageNet dataset by the `resnet50` model in [MindSpore Vision](https://mindspore.cn/vision/docs/en/r1.7/index.html).

The training model is loaded using the `load_checkpoint` interface, which returns a Ditt type, the dictionary's key is the name of each layer of the network, the type is the character Type Str; the value, dictionary value is the parameter value of the network layer, and the type is Parameter.

In the following example, since the number of classification classes of the Resnet50 pre-trained model is 1000, and the number of classification classes of the resnet50 network defined in the example is 2, the fully connected layer parameter in the pre-trained model needs to be deleted.

```python
from mindvision.classification.models import resnet50
from mindspore import load_checkpoint, load_param_into_net
from mindvision.dataset import DownLoad

# Download the pre-trained model for Resnet50
dl = DownLoad()
dl.download_url('https://download.mindspore.cn/vision/classification/resnet50_224.ckpt')
# Define a resnet50 network with a classification class of 2
resnet = resnet50(2)
# Model parameters are saved to the param_dict
param_dict = load_checkpoint("resnet50_224.ckpt")

# Get a list of parameter names for the fully connected layer
param_filter = [x.name for x in resnet.head.get_parameters()]

def filter_ckpt_parameter(origin_dict, param_filter):
    """Delete elements including param_filter parameter names in the origin_dict"""
    for key in list(origin_dict.keys()): # Get all parameter names for the model
        for name in param_filter: # Iterate over the parameter names in the model to be deleted
            if name in key:
                print("Delete parameter from checkpoint:", key)
                del origin_dict[key]
                break

# Delete the full connection layer
filter_ckpt_parameter(param_dict, param_filter)

# Prints the updated model parameters
load_param_into_net(resnet, param_dict)
```

```text
Delete parameter from checkpoint: head.dense.weight
Delete parameter from checkpoint: head.dense.bias
```

## Model Export

MindSpore's `export` can export network models as files in a specified format for inference on other hardware platforms. The main parameters of `export` are as follows:

- `net`: MindSpore network structure.
- `inputs`: The input of the network, the supported input type is Tensor. When there are multiple inputs, they need to be passed in together, such as `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='MINDIR')`.
- `file_name`: Export the file name of the model, if the `file_name` does not contain the corresponding suffix name (such as .mindir), the system will automatically add a suffix to the file name after setting the `file_format`.
- `file_format`: MindSpore currently supports exporting models in "AIR", "ONNX" and "MINDIR" formats.

The following describes the use of `export` to generate corresponding MindIR, AIR and ONNX format files for the resnet50 network and the corresponding CheckPoint format files.

### Export MindIR Model

If you want to perform inference across platforms or hardware (Ascend AI processor, MindSpore on-device, GPU, etc.), you can generate the corresponding MindIR format model file through the network definition and CheckPoint. MindIR format file can be applied to MindSpore Lite. Currently, it supports inference network based on static graph mode. The following is to use the `resnet50` model in MindSpore Vision and the model file resnet50_224.ckpt trained by the model on the ImageNet dataset to export the MindIR format file.

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint
from mindvision.classification.models import resnet50

resnet = resnet50(1000)
load_checkpoint("resnet50_224.ckpt", net=resnet)

input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)

# Export the file resnet50_224.mindir to the current folder
export(resnet, Tensor(input_np), file_name='resnet50_224', file_format='MINDIR')
```

If you wish to save the data preprocess operations into MindIR and use them to perform inference, you can pass the Dataset object into export method:

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from mindspore import export, load_checkpoint
from mindvision.classification.models import resnet50
from mindvision.dataset import DownLoad

def create_dataset_for_renset(path):
    """Create a dataset"""
    data_set = ds.ImageFolderDataset(path)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    data_set = data_set.map(operations=[C.Decode(), C.Resize(256), C.CenterCrop(224),
                                        C.Normalize(mean=mean, std=std), C.HWC2CHW()], input_columns="image")
    data_set = data_set.batch(1)
    return data_set

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip"
path = "./datasets"
# Download and extract the dataset
dl = DownLoad()
dl.download_and_extract_archive(url=dataset_url, download_path=path)
# Load the dataset
path = "./datasets/DogCroissants/val/"
de_dataset = create_dataset_for_renset(path)
# Define the network
resnet = resnet50()

# Load the preprocessing model parameters into the network
load_checkpoint("resnet50_224.ckpt", net=resnet)
# Export a MindIR file with preprocessing information
export(resnet, de_dataset, file_name='resnet50_224', file_format='MINDIR')
```

> - If `file_name` does not contain the ".mindir" suffix, the system will automatically add the ".mindir" suffix to it.

In order to avoid the hardware limitation of Protocol Buffers, when the exported model parameter size exceeds 1G, the framework will save the network structure and parameters separately by default.

- The name of the network structure file ends with the user-specified prefix plus _graph.mindir.
- In the same level directory, there will be a folder with user-specified prefix plus _variables, which stores network parameters.
  And when the parameter's data size exceeds 1T, it will split to another file named data_1, data_2, etc.

Taking the above code as an example, if the parameter size with the model exceeds 1G, the generated directory structure is as follows:

```text
├── resnet50_224_graph.mindir
└── resnet50_224_variables
    ├── data_1
    ├── data_2
    └── data_3
```

### Export AIR Model

If you want to perform inference on the Ascend AI processor, you can also generate the corresponding AIR format model file through the network definition and CheckPoint. The following is to use the `resnet50` model in MindSpore Vision and the model file trained by the model on the ImageNet dataset resnet50_224.ckpt, and export the AIR format file on the Ascend AI processor.

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint
from mindvision.classification.models import resnet50

resnet = resnet50()
# Load parameters into the network
load_checkpoint("resnet50_224.ckpt", net=resnet)
# Network input
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
# Save the resnet50_224.air file to the current directory
export(resnet, Tensor(input_np), file_name='resnet50_224', file_format='AIR')
```

If file_name does not contain the ".air" suffix, the system will automatically add the ".air" suffix to it.

### Export ONNX Model

When you have a CheckPoint file, if you want to do inference on Ascend AI processor, GPU, or CPU, you need to generate ONNX models based on the network and CheckPoint. The following is to use the `resnet50` model in MindSpore Vision and the model file trained by the model on the ImageNet dataset resnet50_224.ckpt, and export the ONNX format file.

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint
from mindvision.classification.models import resnet50

resnet = resnet50()
load_checkpoint("resnet50_224.ckpt", net=resnet)

input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)

# Save the resnet50_224.onnx file to the current directory
export(resnet, Tensor(input_np), file_name='resnet50_224', file_format='ONNX')
```

> - If `file_name` does not contain the ".onnx" suffix, the system will automatically add the ".onnx" suffix to it.
> - Currently, only the ONNX format export of ResNet series networks, YOLOV3, YOLOV4 and BERT are supported.
