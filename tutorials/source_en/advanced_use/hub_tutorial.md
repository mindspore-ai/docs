## Submitting, Loading and Fine-tuning Models using MindSpore Hub

`Ascend` `GPU` `MindSpore Hub` `Model Submission` `Model Loading` `Model Fine-tuning` `Beginner` `Intermediate` `Expert`

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

For algorithm developers who are interested in publishing models into MindSpore Hub, this tutorial introduces the specific steps to submit models using GoogleNet as an example. It also describes how to load/fine-tune MindSpore Hub models for application developers who aim to do inference/transfer learning on new dataset.  In summary, this tutorial helps the algorithm developers submit models efficiently and enables the application developers to perform inference or fine-tuning using MindSpore Hub APIs quickly. 

### How to submit models

We accept publishing models to MindSpore Hub via PR in `hub` repo. Here we use GoogleNet as an example to list the steps of model submission to MindSpore Hub. 

#### Steps

1. Host your pre-trained model in a storage location where we are able to access. 

2. Add a model generation python file called `mindspore_hub_conf.py` in your own repo using this [template](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/googlenet/mindspore_hub_conf.py). 

3. Create a `{model_name}_{model_version}_{dataset}.md` file in `hub/mshub_res/assets` using this [template](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/gpu/0.6/alexnet_v1_cifar10.md). For each pre-trained model, please run the following command to obtain a hash value required at `asset-sha256` of this `.md` file:

   ```python
   cd ../tools
   python get_sha256.py ../googlenet.ckpt
   ```

4. Check the format of the markdown file locally using `hub/mshub_res/tools/md_validator.py` by running the following command:

   ```python
   python md_validator.py ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. Create a PR in `mindspore/hub` repo.

Once your PR is merged into master branch here, your model will show up in [MindSpore Hub Website](https://hub.mindspore.com/mindspore) within 24 hours. For more information, please refer to the [README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md). 

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
from mindspore.dataset.transforms import Compose
from PIL import Image
import mindspore.dataset.vision.py_transforms as py_transforms
import cv2

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    device_id=0)

model = "mindspore/ascend/0.7/googlenet_v1_cifar10"

image = Image.open('cifar10/a.jpg')
transforms = Compose([py_transforms.ToTensor()])

# Initialize the number of classes based on the pre-trained model.
network = mshub.load(model, num_classes=10)
network.set_train(False)
out = network(transforms(image))
```

### Model Fine-tuning

When loading a model with `mindspore_hub.load` API, we can add an extra argument to load the feature extraction part of the model only. So we can easily add new layers to perform transfer learning. *This feature can be found in the related model page when an extra argument (e.g., include_top) has been integrated into the model construction by the algorithm engineer.* 

We use Googlenet as example to illustrate how to load a model trained on ImageNet dataset and then perform transfer learning (re-training) on specific sub-task dataset. The main steps are listed below: 

1. Search the model of interest on [MindSpore Hub Website](https://hub.mindspore.com/mindspore) and get the related `url`. 

2. Load the model from MindSpore Hub using the `url`. *Note that the parameter `include_top` is provided by the model developer*.

   ```python
   import mindspore
   from mindspore import nn
   from mindspore import context
   import mindspore_hub as mshub
   
   context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                       save_graphs=False)
   
   network = mshub.load('mindspore/ascend/0.7/googlenet_v1_cifar10', include_top=False)
   network.set_train(False)
   ```

3. Add a new classification layer into current model architecture.

   ```python
   # Check MindSpore Hub website to conclude that the last output shape is 1024.
   last_channel = 1024
   
   # The number of classes in target task is 26.
   num_classes = 26
   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(True)
   
   train_network = nn.SequentialCell([network, classification_layer])
   ```

4. Define `loss` and `optimizer` for training.

   ```python
   from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
   
   # Wrap the backbone network with loss.
   loss_fn = SoftmaxCrossEntropyWithLogits()
   loss_net = nn.WithLossCell(train_network, loss_fn)
   
   # Create an optimizer.
   optim = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 							Tensor(lr), config.momentum, config.weight_decay)
   
   train_net = nn.TrainOneStepCell(loss_net, optim)
   ```

5. Create dataset and start fine-tuning.

   ```python
   from src.dataset import create_dataset
   from mindspore.train.serialization import _exec_save_checkpoint
   
   dataset = create_dataset("/ssd/data/garbage/train", do_train=True, batch_size=32)
   
   epoch_size = 15
   for epoch in range(epoch_size):
       for i, items in enumerate(dataset):
           data, label = items
           data = mindspore.Tensor(data)
           label = mindspore.Tensor(label)
           
           loss = train_net(data, label)
           print(f"epoch: {epoch}, loss: {loss}")
       # Save the ckpt file for each epoch.
       ckpt_path = f"./ckpt/garbage_finetune_epoch{epoch}.ckpt"
       _exec_save_checkpoint(train_network, ckpt_path)
   ```

6. Eval on test set.

   ```python
   from mindspore.train.serialization import load_checkpoint, load_param_into_net
   
   network = mshub.load('mindspore/ascend/0.7/googlenet_v1_cifar10', include_top=False)
   train_network = nn.SequentialCell([network, nn.Dense(last_channel, num_classes)])
   
   # Load a pre-trained ckpt file.
   ckpt_path = "./ckpt/garbage_finetune_epoch15.ckpt"
   trained_ckpt = load_checkpoint(ckpt_path)
   load_param_into_net(train_network, trained_ckpt)
   
   # Define loss and create model.
   loss = SoftmaxCrossEntropyWithLogits()
   model = Model(network, loss_fn=loss, metrics={'acc'})
   
   eval_dataset = create_dataset("/ssd/data/garbage/train", do_train=False, 
                                 batch_size=32)
   
   res = model.eval(eval_dataset)
   print("result:", res, "ckpt=", ckpt_path)
   ```

