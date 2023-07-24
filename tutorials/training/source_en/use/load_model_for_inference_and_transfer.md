# Loading a Model for Inference and Transfer Learning

`Linux` `Ascend` `GPU` `CPU` `Model Loading` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/training/source_en/use/load_model_for_inference_and_transfer.md)

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
   from mindspore import context, Tensor, nn, Model
   from mindspore import dtype as mstype
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

3. After loading the model, you can use MindSpore to do inference. You can refer to [Multi-Platform Inference Overview](https://www.mindspore.cn/tutorial/inference/en/r1.2/multi_platform_inference.html).

### For Transfer Training

When loading a model with `mindspore_hub.load` API, we can add an extra argument to load the feature extraction part of the model only. So we can easily add new layers to perform transfer learning. This feature can be found in the related model page when an extra argument (e.g., include_top) has been integrated into the model construction by the model developer. The value of `include_top` is True or False, indicating whether to keep the top layer in the fully-connected network.

We use [MobileNetV2](https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo/official/cv/mobilenetv2) as an example to illustrate how to load a model trained on the ImageNet dataset and then perform transfer learning (re-training) on a specific sub-task dataset. The main steps are listed below:

1. Search the model of interest on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub/) and get the related `url`.

2. Load the model from MindSpore Hub using the `url`. Note that the parameter `include_top` is provided by the model developer.

   ```python
   import os
   import mindspore_hub as mshub
   import mindspore
   from mindspore import context, Tensor, nn
   from mindspore.nn import Momentum
   from mindspore.train.serialization import save_checkpoint, load_checkpoint,load_param_into_net
   from mindspore import ops
   import mindspore.dataset as ds
   import mindspore.dataset.transforms.c_transforms as C2
   import mindspore.dataset.vision.c_transforms as C
   from mindspore import dtype as mstype
   from mindspore import Model
   context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

   model = "mindspore/ascend/1.0/mobilenetv2_v1.0_openimage"
   network = mshub.load(model, num_classes=500, include_top=False, activation="Sigmoid")
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

   # Check MindSpore Hub website to conclude that the last output shape is 1280.
   last_channel = 1280

   # The number of classes in target task is 10.
   num_classes = 10

   reducemean_flatten = ReduceMeanFlatten()

   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(True)

   train_network = nn.SequentialCell([network, reducemean_flatten, classification_layer])
   ```

4. Define `dataset_loader`.

   As shown below， the new dataset used for fine-tuning is the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). It is noted here we need to download the `binary version` dataset. After downloading and decompression, the following code can be used for data loading and processing. It is noted the `dataset_path` is the path to the dataset and should be given by the user.

   ```python
   def create_cifar10dataset(dataset_path, batch_size, do_train):
       if do_train:
           usage, shuffle = "train", True
       else:
           usage, shuffle = "test", False

       data_set = ds.Cifar10Dataset(dataset_dir=dataset_path, usage=usage, shuffle=True)

       # define map operations
       trans = [C.Resize((256, 256))]
       if do_train:
           trans += [
               C.RandomHorizontalFlip(prob=0.5),
           ]

       trans += [
           C.Rescale(1.0 / 255.0, 0.0),
           C.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
           C.HWC2CHW()
       ]

       type_cast_op = C2.TypeCast(mstype.int32)

       data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
       data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

       # apply batch operations
       data_set = data_set.batch(batch_size, drop_remainder=True)
       return data_set

   # Create Dataset
   dataset_path = "/path_to_dataset/cifar-10-batches-bin"
   dataset = create_cifar10dataset(dataset_path, batch_size=32, do_train=True)
   ```

5. Define `loss`, `optimizer` and `learning rate`.

   ```python
   def generate_steps_lr(lr_init, steps_per_epoch, total_epochs):
       total_steps = total_epochs * steps_per_epoch
       decay_epoch_index = [0.3*total_steps, 0.6*total_steps, 0.8*total_steps]
       lr_each_step = []
       for i in range(total_steps):
           if i < decay_epoch_index[0]:
               lr = lr_init
           elif i < decay_epoch_index[1]:
               lr = lr_init * 0.1
           elif i < decay_epoch_index[2]:
               lr = lr_init * 0.01
           else:
               lr = lr_init * 0.001
           lr_each_step.append(lr)
       return lr_each_step

   # Set epoch size
   epoch_size = 60

   # Wrap the backbone network with loss.
   loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
   loss_net = nn.WithLossCell(train_network, loss_fn)
   steps_per_epoch = dataset.get_dataset_size()
   lr = generate_steps_lr(lr_init=0.01, steps_per_epoch=steps_per_epoch, total_epochs=epoch_size)
   # Create an optimizer.
   optim = Momentum(filter(lambda x: x.requires_grad, classification_layer.get_parameters()), Tensor(lr, mindspore.float32), 0.9, 4e-5)
   train_net = nn.TrainOneStepCell(loss_net, optim)
   ```

6. Start fine-tuning.

   ```python
   for epoch in range(epoch_size):
       for i, items in enumerate(dataset):
           data, label = items
           data = mindspore.Tensor(data)
           label = mindspore.Tensor(label)

           loss = train_net(data, label)
           print(f"epoch: {epoch}/{epoch_size}, loss: {loss}")
       # Save the ckpt file for each epoch.
       if not os.path.exists('ckpt'):
          os.mkdir('ckpt')
       ckpt_path = f"./ckpt/cifar10_finetune_epoch{epoch}.ckpt"
       save_checkpoint(train_network, ckpt_path)
   ```

6. Eval on test set.

   ```python
   model = "mindspore/ascend/1.0/mobilenetv2_v1.0_openimage"

   network = mshub.load(model, num_classes=500, pretrained=True, include_top=False, activation="Sigmoid")
   network.set_train(False)
   reducemean_flatten = ReduceMeanFlatten()
   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(False)
   softmax = nn.Softmax()
   network = nn.SequentialCell([network, reducemean_flatten, classification_layer, softmax])

   # Load a pre-trained ckpt file.
   ckpt_path = "./ckpt/cifar10_finetune_epoch59.ckpt"
   trained_ckpt = load_checkpoint(ckpt_path)
   load_param_into_net(classification_layer, trained_ckpt)

   loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

   # Define loss and create model.
   eval_dataset = create_cifar10dataset(dataset_path, batch_size=32, do_train=False)
   eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
   model = Model(network, loss_fn=loss, optimizer=None, metrics=eval_metrics)
   metrics = model.eval(eval_dataset)
   print("metric: ", metrics)
   ```
