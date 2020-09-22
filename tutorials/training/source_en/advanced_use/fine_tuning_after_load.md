# Fine-tuning Models using MindSpore Hub

`Linux` `Ascend` `GPU` `Model Loading` `Model Fine-tuning` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Fine-tuning Models using MindSpore Hub](#fine-tuning-models-using-mindspore-hub)
    - [Overview](#overview)
    - [Model Fine-tuning](#model-fine-tuning)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/fine_tuning_after_load.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Fine-tuning is TODO.

This tutorial describes how to fine-tune MindSpore Hub models for application developers who aim to do inference/transfer learning on new dataset, it helps users to perform inference or fine-tuning using MindSpore Hub APIs quickly. 

## Model Fine-tuning

When loading a model with `mindspore_hub.load` API, we can add an extra argument to load the feature extraction part of the model only. So we can easily add new layers to perform transfer learning. This feature can be found in the related model page when an extra argument (e.g., include_top) has been integrated into the model construction by the model developer. The value of `include_top` is True or False, indicating whether to keep the top layer in the fully-connected network. 

We use GoogleNet as example to illustrate how to load a model trained on ImageNet dataset and then perform transfer learning (re-training) on specific sub-task dataset. The main steps are listed below: 

1. Search the model of interest on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub/) and get the related `url`. 

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