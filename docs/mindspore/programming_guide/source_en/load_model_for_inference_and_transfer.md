# Loading a Model for Inference and Transfer Learning

`Linux` `Ascend` `GPU` `CPU` `Model Loading` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/load_model_for_inference_and_transfer.md)

## Overview

CheckPoints which are saved locally during model training, they are used for inference and transfer training.

The following uses examples to describe how to load models from local.

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
opt = Momentum(resnet.trainable_params(), 0.01, 0.9)
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into optimizer
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

The `load_checkpoint` method returns a parameter dictionary and then the `load_param_into_net` method loads parameters in the parameter dictionary to the network or optimizer.
