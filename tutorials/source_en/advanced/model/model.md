# Basic Usage of Models

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/advanced/model/model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

Generally, defining a training and evaluation network and running it directly can meet basic requirements.

On the one hand, `Model` can simplify code to some extent. For example, you do not need to manually traverse datasets. If you do not need to customize `nn.TrainOneStepCell`, you can use `Model` to automatically build a training network. You can use the `eval` API of `Model` to evaluate the model and directly output the evaluation result. You do not need to manually invoke the `clear`, `update`, and `eval` functions of evaluation metrics.

On the other hand, `Model` provides many high-level functions, such as data offloading and mixed precision. Without the help of `Model`, it takes a long time to customize these functions by referring to `Model`.

The following describes MindSpore models and how to use `Model` for model training, evaluation, and inference.

![model](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_en/advanced/model/images/model.png)

```python
import mindspore
from mindspore import nn
from mindspore.dataset import MnistDataset, vision, transforms
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
```

## Introduction to Model

[Model](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.Model.html#mindspore.Model) is a high-level API provided by MindSpore for model training, evaluation, and inference. The common parameters of the API are as follows:

- `network`: neural network used for training or inference.
- `loss_fn`: used loss function.
- `optimizer`: used optimizer.
- `metrics`: evaluation function used for model evaluation.
- `eval_network`: network used for model evaluation. If the network is not defined, `Model` uses `network` and `loss_fn` for encapsulation.

`Model` provides the following APIs for model training, evaluation, and inference:

- `fit`: used to evaluate the model during training.
- `train`: used for model training on the training set.
- `eval`: used to evaluate the model on the evaluation set.
- `predict`: performs inference on a group of input data and outputs the prediction result.

### Using the Model API

For a neural network in a simple scenario, you can specify the feedforward network ``network``, loss function ``loss_fn``, optimizer ``optimizer``,
and evaluation function ``metrics`` when defining ``Model``.

## Download and Process Dataset

```python
# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip")


def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_dataset = datapipe('MNIST_Data/train', 64)
test_dataset = datapipe('MNIST_Data/test', 64)
```

## Define Model

```python
# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
```

## Define loss function and optimizer

To train neural network model, loss function and optimizer function need to be defined.

- The loss function here uses ``CrossEntropy Loss`` .
- The optimizer uses SGD here.

```python
# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)
```

## Train and Save Model

Before starting the training, MindSpot needs to state in advance whether the network model needs to save the intermediate process and results
during the training process. Therefore, ``ModelCheckpoint`` is used to save the network model and parameters for subsequent fine tuning.

```python
steps_per_epoch = train_dataset.get_dataset_size()
config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)

ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)
loss_callback = LossMonitor(steps_per_epoch)
```

``model.fit`` provided by MindSpore can facilitate network training, and ``LossMonitor`` can monitor the change of loss value during training.

```python
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})

trainer.fit(10, train_dataset, test_dataset, callbacks=[ckpt_callback, loss_callback])
```

```text
epoch: 1 step: 937, loss is 0.5313149094581604
Eval result: epoch 1, metrics: {'accuracy': 0.8557692307692307}
epoch: 2 step: 937, loss is 0.2875961363315582
Eval result: epoch 2, metrics: {'accuracy': 0.9007411858974359}
epoch: 3 step: 937, loss is 0.19009104371070862
Eval result: epoch 3, metrics: {'accuracy': 0.9191706730769231}
epoch: 4 step: 937, loss is 0.24231787025928497
Eval result: epoch 4, metrics: {'accuracy': 0.9296875}
epoch: 5 step: 937, loss is 0.16016671061515808
Eval result: epoch 5, metrics: {'accuracy': 0.9386017628205128}
epoch: 6 step: 937, loss is 0.4830142855644226
Eval result: epoch 6, metrics: {'accuracy': 0.9444110576923077}
epoch: 7 step: 937, loss is 0.20778779685497284
Eval result: epoch 7, metrics: {'accuracy': 0.9508213141025641}
epoch: 8 step: 937, loss is 0.22020074725151062
Eval result: epoch 8, metrics: {'accuracy': 0.9540264423076923}
epoch: 9 step: 937, loss is 0.15951070189476013
Eval result: epoch 9, metrics: {'accuracy': 0.9575320512820513}
epoch: 10 step: 937, loss is 0.11161471903324127
Eval result: epoch 10, metrics: {'accuracy': 0.9608373397435898}
```

During training, the loss value will be printed, and the loss value will fluctuate, but in general, the loss value will gradually decrease and
the accuracy will gradually improve. The loss values run by each person are random and not necessarily identical.

The results obtained by running the test data set of the model verify the generalization ability of the model:

- Use ``model.eval`` to read in the test data set.
- Use the saved model parameters for reasoning.

```python
acc = trainer.eval(test_dataset)
acc
```

```text
{'accuracy': 0.9641}
```

The model precision data can be seen from the print information. In the example, the precision data reaches more than 95%, and the model quality
is good. As the number of network iterations increases, the accuracy of the model will be further improved.