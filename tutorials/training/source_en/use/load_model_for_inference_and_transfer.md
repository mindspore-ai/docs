# Loading a Model for Inference and Transfer Learning

`Linux` `Ascend` `GPU` `CPU` `Model Loading` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/use/load_model_for_inference_and_transfer.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

CheckPoints which are saved locally during model training, or download from [MindSpore Hub](https://www.mindspore.cn/resources/hub/) are used for inference and transfer training.

The following uses examples to describe how to load models from local and MindSpore Hub.

## Loading the local Model

After saving CheckPoint files, you can load parameters.

### For Inference Validation

In inference-only scenarios, use `load_checkpoint` to directly load parameters to the network for subsequent inference validation.

The sample code is as follows:

```python
resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
dateset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1) # define the test dataset
loss = CrossEntropyLoss()
model = Model(resnet, loss, metrics={"accuracy"})
acc = model.eval(dataset_eval)
```

The `load_checkpoint` method loads network parameters in the parameter file to the model. After the loading, parameters in the network are those saved in CheckPoints.
The `eval` method validates the accuracy of the trained model.

### For Transfer Training

In the retraining and fine-tuning scenarios for task interruption, you can load network parameters and optimizer parameters to the model.

The sample code is as follows:

```python
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
resnet = ResNet50()
opt = Momentum()
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into optimizer
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

The `load_checkpoint` method returns a parameter dictionary and then the `load_param_into_net` method loads parameters in the parameter dictionary to the network or optimizer.

## Loading the Model from Hub

### For Inference Validation

`mindspore_hub.load` API is used to load the pre-trained model in a single line of code. The main process of model loading is as follows:

1. Search the model of interest on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub).

   For example, if you aim to perform image classification on CIFAR-10 dataset using GoogleNet, please search on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub) with the keyword `GoogleNet`. Then all related models will be returned.  Once you enter into the related model page, you can get the website `url`.

2. Complete the task of loading model using `url` , as shown in the example below:

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

3. After loading the model, you can use MindSpore to do inference. You can refer to [here](https://www.mindspore.cn/tutorial/inference/en/r1.0/multi_platform_inference.html).

### For Transfer Training

When loading a model with `mindspore_hub.load` API, we can add an extra argument to load the feature extraction part of the model only. So we can easily add new layers to perform transfer learning. This feature can be found in the related model page when an extra argument (e.g., include_top) has been integrated into the model construction by the model developer. The value of `include_top` is True or False, indicating whether to keep the top layer in the fully-connected network.

We use GoogleNet as example to illustrate how to load a model trained on ImageNet dataset and then perform transfer learning (re-training) on specific sub-task dataset. The main steps are listed below:

1. Search the model of interest on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub/) and get the related `url`.

2. Load the model from MindSpore Hub using the `url`. Note that the parameter `include_top` is provided by the model developer.

   ```python
   import mindspore
   from mindspore import nn, context, Tensor
   from mindspore.train.serialization import save_checkpoint
   from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
   import mindspore.ops as ops
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
            self.mean = ops.ReduceMean(keep_dims=True)
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
