## Submitting, Loading and Fine-tuning Models using MindSpore Hub

`Linux` `Ascend` `GPU` `MindSpore Hub` `Model Submission` `Model Loading` `Model Fine-tuning` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Submitting, Loading and Fine-tuning Models using MindSpore Hub](#submitting-loading-and-fine-tuning-models-using-mindspore-hub)
  - [Overview](#overview)
  - [How to submit models](#how-to-submit-models)
    - [Steps](#steps)
  - [How to load models](#how-to-load-models)
  - [Model Fine-tuning](#model-fine-tuning)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced_use/hub_tutorial.md" target="_blank"><img src="../_static/logo_source.png"></a>

### Overview

MindSpore Hub is a pre-trained model application tool of the MindSpore ecosystem, which serves as a channel for model developers and application developers. It not only provides model developers with a convenient and fast channel for model submission, but also provides application developers with simple model loading and fine-tuning APIs. For model developers who are interested in publishing models into MindSpore Hub, this tutorial introduces the specific steps to submit models using GoogleNet as an example. It also describes how to load/fine-tune MindSpore Hub models for application developers who aim to do inference/transfer learning on new dataset.  In summary, this tutorial helps the model developers submit models efficiently and enables the application developers to perform inference or fine-tuning using MindSpore Hub APIs quickly. 

### How to submit models

We accept publishing models to MindSpore Hub via PR in [hub](https://gitee.com/mindspore/hub) repo. Here we use GoogleNet as an example to list the steps of model submission to MindSpore Hub. 

#### Steps

1. Host your pre-trained model in a storage location where we are able to access. 

2. Add a model generation python file called `mindspore_hub_conf.py` in your own repo using this [template](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/googlenet/mindspore_hub_conf.py). The location of the `mindspore_hub_conf.py` file is shown below:

   ```shell script
   googlenet
   ├── src
   │   ├── googlenet.py
   ├── script
   │   ├── run_train.sh
   ├── train.py
   ├── test.py
   ├── mindspore_hub_conf.py
   ```

3. Create a `{model_name}_{model_version}_{dataset}.md` file in `hub/mshub_res/assets/mindspore/ascend/0.7` using this [template](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md). Here `ascend` refers to the hardware platform for the pre-trained model, and `0.7` indicates the MindSpore version. The structure of the `hub/mshub_res` folder is as follows：

   ```shell script
   hub
   ├── mshub_res
   │   ├── assets
   │       ├── mindspore
   |           ├── gpu
   |               ├── 0.7
   |           ├── ascend
   |               ├── 0.7 
   |                   ├── googlenet_v1_cifar10.md
   │   ├── tools
   |       ├── md_validator.py
   |       └── md_validator.py 
   ```
   
   Note that it is required to fill in the `{model_name}_{model_version}_{dataset}.md` template by providing `file-format`、`asset-link` and `asset-sha256` below, which refers to the model file format, model storage location from step 1 and model hash value, respectively. The MindSpore Hub supports multiple model file formats including [MindSpore CKPT](https://www.mindspore.cn/tutorial/en/master/use/saving_and_loading_model_parameters.html#checkpoint-configuration-policies), [AIR](https://www.mindspore.cn/tutorial/en/master/use/multi_platform_inference.html), [MindIR](https://www.mindspore.cn/tutorial/en/master/use/saving_and_loading_model_parameters.html#export-mindir-model), [ONNX](https://www.mindspore.cn/tutorial/en/master/use/multi_platform_inference.html) and [MSLite](https://www.mindspore.cn/lite/tutorial/en/master/use/converter_tool.html).

     ```shell script
   file-format: ckpt  
   asset-link: https://download.mindspore.cn/model_zoo/official/cv/googlenet/goolenet_ascend_0.2.0_cifar10_official_classification_20200713/googlenet.ckpt  
   asset-sha256: 114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7
   ```
   For each pre-trained model, please run the following command to obtain a hash value required at `asset-sha256` of this `.md` file. Here the pre-trained model `googlenet.ckpt` is accessed from the storage location in step 1 and then saved in `tools` folder. The output hash value is: `114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7`.

   ```python
   cd ../tools
   python get_sha256.py ../googlenet.ckpt
   ```

4. Check the format of the markdown file locally using `hub/mshub_res/tools/md_validator.py` by running the following command. The output is `All Passed`，which indicates that the format and content of the `.md` file meets the requirements.

   ```python
   python md_validator.py ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. Create a PR in `mindspore/hub` repo. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more information about creating a PR. 

Once your PR is merged into master branch here, your model will show up in [MindSpore Hub Website](https://hub.mindspore.com/mindspore) within 24 hours. Please refer to [README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md) for more information about model submission. 

### How to load models

`mindspore_hub.load` API is used to load the pre-trained model in a single line of code. The main process of model loading is as follows:

- Search the model of interest on [MindSpore Hub Website](https://hub.mindspore.com/mindspore).

  For example, if you aim to perform image classification on CIFAR-10 dataset using GoogleNet, please search on [MindSpore Hub Website](https://hub.mindspore.com/mindspore) with the keyword `GoogleNet`. Then all related models will be returned.  Once you enter into the related model page, you can get the website `url`.

- Complete the task of loading model using `url` , as shown in the example below:

  ```python
  
  import mindspore_hub as mshub
  import mindspore
  from mindspore import context, Tensor, nn
  from mindspore.train.model import Model
  from mindspore.common import dtype as mstype
  import mindspore.dataset.vision.py_transforms as py_transforms
  
  context.set_context(mode=context.GRAPH_MODE,
                      device_target="Ascend",
                      device_id=0)
  
  model = "mindspore/ascend/0.7/googlenet_v1_cifar10"
  
  # Initialize the number of classes based on the pre-trained model.
  network = mshub.load(model, num_classes=10)
  network.set_train(False)
  
  # ...
  
  ```
- After loading the model, you can use MindSpore to do inference. You can refer to [here](https://www.mindspore.cn/tutorial/en/master/use/multi_platform_inference.html).

### Model Fine-tuning

When loading a model with `mindspore_hub.load` API, we can add an extra argument to load the feature extraction part of the model only. So we can easily add new layers to perform transfer learning. This feature can be found in the related model page when an extra argument (e.g., include_top) has been integrated into the model construction by the model developer. The value of `include_top` is True or False, indicating whether to keep the top layer in the fully-connected network. 

We use GoogleNet as example to illustrate how to load a model trained on ImageNet dataset and then perform transfer learning (re-training) on specific sub-task dataset. The main steps are listed below: 

1. Search the model of interest on [MindSpore Hub Website](https://hub.mindspore.com/mindspore) and get the related `url`. 

2. Load the model from MindSpore Hub using the `url`. Note that the parameter `include_top` is provided by the model developer.

   ```python
   import mindspore
   from mindspore import nn, context, Tensor
   from mindpsore.train.serialization import save_checkpoint
   from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
   from mindspore.ops import operations as P
   from mindspore.nn import Momentum
   
   import math
   import numpy as np
   
   import mindspore_hub as mshub
   from src.dataset import create_dataset
   
   context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                       save_graphs=False)
   model_url = "mindspore/ascend/0.7/googlenet_v1_cifar10"
   network = mshub.load(model_url, include_top=False, num_classes=1000)
   network.set_train(False)
   ```

3. Add a new classification layer into current model architecture.

   ```python
   class ReduceMeanFlatten(nn.Cell):
       def __init__(self):
           super(ReduceMeanFlatten, self).__init__()
           self.mean = P.ReduceMean(keep_dims=True)
           self.flatten = nn.Flatten()
       
       def construct(self, x):
           x = self.mean(x, (2, 3))
           x = self.flatten(x)
           return x
   
   # Check MindSpore Hub website to conclude that the last output shape is 1024.
   last_channel = 1024
   
   # The number of classes in target task is 26.
   num_classes = 26
   
   reducemean_flatten = ReduceMeanFlatten()
   
   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(True)
   
   train_network = nn.SequentialCell([network, reducemean_flatten, classification_layer])
   ```

4. Define `loss` and `optimizer` for training.

   ```python
   epoch_size = 60
   
   # Wrap the backbone network with loss.
   loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
   loss_net = nn.WithLossCell(train_network, loss_fn)
   
   lr = get_lr(global_step=0,
               lr_init=0,
               lr_max=0.05,
               lr_end=0.001,
               warmup_epochs=5,
               total_epochs=epoch_size)
   
   # Create an optimizer.
   optim = Momentum(filter(lambda x: x.requires_grad, loss_net.get_parameters()), Tensor(lr), 0.9, 4e-5)
   train_net = nn.TrainOneStepCell(loss_net, optim)
   ```

5. Create dataset and start fine-tuning. As is shown below, the new dataset used for fine-tuning is the garbage classification data located at `/ssd/data/garbage/train` folder.

   ```python
   dataset = create_dataset("/ssd/data/garbage/train",
                            do_train=True,
                            batch_size=32,
                            platform="Ascend",
                            repeat_num=1)

   for epoch in range(epoch_size):
       for i, items in enumerate(dataset):
           data, label = items
           data = mindspore.Tensor(data)
           label = mindspore.Tensor(label)
           
           loss = train_net(data, label)
           print(f"epoch: {epoch}/{epoch_size}, loss: {loss}")
       # Save the ckpt file for each epoch.
       ckpt_path = f"./ckpt/garbage_finetune_epoch{epoch}.ckpt"
       save_checkpoint(train_network, ckpt_path)
   ```

6. Eval on test set.

   ```python
   from mindspore.train.serialization import load_checkpoint, load_param_into_net
   
   network = mshub.load('mindspore/ascend/0.7/googlenet_v1_cifar10', pretrained=False,
                        include_top=False, num_classes=1000)
   
   reducemean_flatten = ReduceMeanFlatten()
   
   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(False)
   softmax = nn.Softmax()
   network = nn.SequentialCell([network, reducemean_flatten, 
                               classification_layer, softmax])
   
   # Load a pre-trained ckpt file.
   ckpt_path = "./ckpt/garbage_finetune_epoch59.ckpt"
   trained_ckpt = load_checkpoint(ckpt_path)
   load_param_into_net(network, trained_ckpt)
   
   # Define loss and create model.
   model = Model(network, metrics={'acc'}, eval_network=network)
   
   eval_dataset = create_dataset("/ssd/data/garbage/test",
                            do_train=True,
                            batch_size=32,
                            platform="Ascend",
                            repeat_num=1)
   
   res = model.eval(eval_dataset)
   print("result:", res, "ckpt=", ckpt_path)
   ```