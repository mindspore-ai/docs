# Basic Use of Model

Translator: [Soleil](https://gitee.com/deng-zhihua)

`Ascend` `GPU` `CPU` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_en/model_use_guide.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

BUILD THE NETWORK of [Programming Guide](https://www.mindspore.cn/docs/programming_guide/en/r1.6/index.html)describes how to define the forward network, loss function and optimizer. In addition, it shows how to encapsulate these structures into training and evaluating networks and execute them. On this basis, this document is about how to use the high-level API `Model` for training and evaluating models.

In general, it is sufficient for basic needs when you can define training and evaluating networks and run them directly. However, it is still recommended to train and evaluate models by `Model`. On the one hand, `Model` can simplify the code in some degree. For example, there is no need to manually traverse the dataset. In the case without the need to customize `TrainOneStepCell`, `Model` can be used to automatically build the training network. The `eval` interface of `Model` can be used for model evaluation with direct output of evaluation results, which is not necessary to invoke the evaluation indicators' functions such as `clear`, `update`, `eval`. On the other hand, `Model` provides many high-level functions, such as data sinking and mixing accuracy. Without the help of `Model`, it would take more time to use these functions by imitating `Model` for customization.

This document starts with a basic introduction of Model, and then focuses on how to use `Model` for Model Training, Evaluation and Inference.

> In the following example, the parameter initialization uses random values, which may result in different outputs from local execution. If you need a stable output of fixed values, you can set a fixed random seed. The setting method can be referred to [mindspore.set_seed()](https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore/mindspore.set_seed.html).

## Basic Introduction of Model

`Model` is a high-level API for model training provided by MindSpore, which can be used for model training, evaluation and inference.

The `Model` contains the following input parameters:

- network (Cell)：In general, it is a forward network which inputs data and labels, and outputs predicted values.

- loss_fn (Cell)：The loss function used.

- optimizer (Cell)：The optimizer used.

- metrics (set)：Evaluation metrics used for model evaluation. The default value `None` will be used when there is no need for model evaluation.

- eval_network (Cell)：The network used for model evaluation which does not need to be specified in some simple cases.

- eval_indexes (List)：It is used to indicate the meaning of the evaluation network output in combination with `eval_network`. The function of this parameter can be replaced by `set_indexes` of `nn.Metric`. It is recommenced to use `set_indexes`.

- amp_level (str)：It is used to specify the mixed accuracy level.

- kwargs：It can configure overflow detection and mixed accuracy policies.

`Model` provides the following interfaces for model training, evaluation and inference:

- train：It is used for model training on the training set.

- eval：It is used for model evaluation on the validation set.

- predict：It is used to inference over the input dataset and output the prediction result.

## Model Training, Evaluation and Inference

For neural networks in simple scenarios, the forward network `network`, loss function `loss_fn`, optimizer `optimizer` and evaluation metrics `metrics` can be specified during defining `Model`. In this case, Model will use `network` as the inference network and build the training network using `nn.WithLossCell` and `nn.TrainOneStepCell` as well as build the evaluation network using `nn.WithEvalCell`.

Take the linear regression used in [Build Training and Evaluating Network](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/train_and_eval.html) as an example:

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

net = LinearNet()
# Set Loss Function
crit = nn.MSELoss()
# Set Optimizer
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# Set Evaluation Metrics
metrics = {"mae"}
```

[Build Training and Evaluating Network](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/train_and_eval.html) describes the way to build and directly run training and evaluating networks via `nn. WithLossCell`, `nn.TrainOneStepCell` and `nn.WithEvalCell`. When using `Model`, there is no need to build the training and evaluating networks manually. You can use the following way to define `Model` and invoke `train` and `eval` interfaces to achieve the same effect.

Create training and validation sets:

```python
import numpy as np
import mindspore.dataset as ds

def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16):
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset

# Create Dataset
train_dataset = create_dataset(num_data=160)
eval_dataset = create_dataset(num_data=80)
```

Define Model and perform training, and check the value of the loss function during training by the `LossMonitor` callback function:

```python
from mindspore import Model
from mindspore.train.callback import LossMonitor

model = Model(network=net, loss_fn=crit, optimizer=opt, metrics=metrics)
# Acquire Training Process Data
epochs = 2
model.train(epochs, train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

The implementation results are as follows:

```text
epoch: 1 step: 1, loss is 158.6485
epoch: 1 step: 2, loss is 56.015274
epoch: 1 step: 3, loss is 22.507223
epoch: 1 step: 4, loss is 29.29523
epoch: 1 step: 5, loss is 54.613194
epoch: 1 step: 6, loss is 119.0715
epoch: 1 step: 7, loss is 47.707245
epoch: 1 step: 8, loss is 6.823062
epoch: 1 step: 9, loss is 12.838973
epoch: 1 step: 10, loss is 24.879482
epoch: 2 step: 1, loss is 38.01019
epoch: 2 step: 2, loss is 34.66765
epoch: 2 step: 3, loss is 13.370583
epoch: 2 step: 4, loss is 3.0936844
epoch: 2 step: 5, loss is 6.6003437
epoch: 2 step: 6, loss is 19.703354
epoch: 2 step: 7, loss is 28.276491
epoch: 2 step: 8, loss is 10.402792
epoch: 2 step: 9, loss is 6.908296
epoch: 2 step: 10, loss is 1.5971221
```

Perform model evaluation and obtain the results:

```python
eval_result = model.eval(eval_dataset)
print(eval_result)
```

The implementation results are as follows:

```text
{'mae': 2.4565244197845457}
```

Inference using `predict`:

```python
for d in eval_dataset.create_dict_iterator():
    data = d["data"]
    break

output = model.predict(data)
print(output)
```

The implementation results are as follows:

```text
[[ 13.330149  ]
 [ -3.380001  ]
 [ 11.5734005 ]
 [ -0.84721684]
 [ 11.391014  ]
 [ -9.029837  ]
 [  1.1881653 ]
 [  2.1025467 ]
 [ 13.401606  ]
 [  1.8194647 ]
 [  8.862836  ]
 [ 14.427877  ]
 [  4.330497  ]
 [-12.431898  ]
 [ -4.5104184 ]
 [  9.439548  ]]
```

In general, post-processing on the inference results is required to obtain more intuitive inference results.

Compared to building the network and then running it directly, there is no need for `set_train` to configure the execution patterns of the network structure when using Model for model training, inference and evaluation.

## Model Applications for Custom Scenarios

As already mentioned in [Loss Function](https://www.mindspore.cn/docs/programming_guide/en/r1.6/loss.html) and [Build Training and Evaluating Network](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/train_and_eval.html), the network encapsulation functions `nn.WithLossCell`, `nn.TrainOneStepCell` and `nn.WithEvalCell` provided by MindSpore are not applicable to all scenarios which means we often need to customize the encapsulation method of the network in real scenarios. In such cases it is obviously not reasonable for `Model` to use these encapsulation functions to encapsulate automatically. The next section will introduce how to properly use `Model` in these cases.

### Connect Forward Network with Loss Function Manually

In scenarios with multiple data or multiple labels, you can manually link the forward network with the custom loss function as the `network` of the `model`, with the default value of `None` for `loss_fn`. Then, `model` will directly use `nn.TrainOneStepCell` to form `network` and `optimizer` into a training network without going through `nn.WithLossCell`.

The following example is from the `Loss Function` <https://www.mindspore.cn/docs/programming_guide/en/r1.6/loss.html>:

1. Define Multi-Label Datasets

    ```python
    import numpy as np
    import mindspore.dataset as ds

    def get_multilabel_data(num, w=2.0, b=3.0):
        for _ in range(num):
            x = np.random.uniform(-10.0, 10.0)
            noise1 = np.random.normal(0, 1)
            noise2 = np.random.normal(-1, 1)
            y1 = x * w + b + noise1
            y2 = x * w + b + noise2
            yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)

    def create_multilabel_dataset(num_data, batch_size=16):
        dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
        dataset = dataset.batch(batch_size)
        return dataset
    ```

2. Customized Multi-Label Loss Function

    ```python
    import mindspore.ops as ops
    from mindspore.nn import LossBase

    class L1LossForMultiLabel(LossBase):
        def __init__(self, reduction="mean"):
            super(L1LossForMultiLabel, self).__init__(reduction)
            self.abs = ops.Abs()

        def construct(self, base, target1, target2):
            x1 = self.abs(base - target1)
            x2 = self.abs(base - target2)
            return self.get_loss(x1)/2 + self.get_loss(x2)/2
    ```

3. Connect the forward network and the loss function, where `net` uses `LinearNet` defined in the previous section

    ```python
    import mindspore.nn as nn

    class CustomWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(CustomWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label1, label2):
            output = self._backbone(data)
            return self._loss_fn(output, label1, label2)
    net = LinearNet()
    loss = L1LossForMultiLabel()
    loss_net = CustomWithLossCell(net, loss)
    ```

4. Define Model and Perform Training

    ```python
    from mindspore.train.callback import LossMonitor
    from mindspore import Model

    opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
    model = Model(network=loss_net, optimizer=opt)
    multi_train_dataset = create_multilabel_dataset(num_data=160)
    model.train(epoch=1, train_dataset=multi_train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)
    ```

    The implementation results are as follows:

    ```text
    epoch: 1 step: 1, loss is 2.7395597
    epoch: 1 step: 2, loss is 3.730921
    epoch: 1 step: 3, loss is 6.393111
    epoch: 1 step: 4, loss is 5.684395
    epoch: 1 step: 5, loss is 6.089678
    epoch: 1 step: 6, loss is 8.953241
    epoch: 1 step: 7, loss is 9.357056
    epoch: 1 step: 8, loss is 8.601417
    epoch: 1 step: 9, loss is 9.339062
    epoch: 1 step: 10, loss is 7.6557174
    ```

5. Model Evaluation

    `Model` uses `nn.WithEvalCell` to build the evaluation network by default, but it is also necessary to build the evaluating network manually when the demand is not satisfied, such as a typical case with multiple data and multiple labels. `Model` provides `eval_network` for setting up a custom evaluating network. The manual construction of the evaluating network is as follows:

    Encapsulation method for the custom evaluation network：

    ```python
    import mindspore.nn as nn

    class CustomWithEvalCell(nn.Cell):
        def __init__(self, network):
            super(CustomWithEvalCell, self).__init__(auto_prefix=False)
            self.network = network

        def construct(self, data, label1, label2):
            output = self.network(data)
            return output, label1, label2
    ```

    Build the Evaluating Network manually：

    ```python
    eval_net = CustomWithEvalCell(net)
    ```

    Use Model for model evaluation：

    ```python
    from mindspore.train.callback import LossMonitor
    from mindspore import Model

    mae1 = nn.MAE()
    mae2 = nn.MAE()
    mae1.set_indexes([0, 1])
    mae2.set_indexes([0, 2])

    model = Model(network=loss_net, optimizer=opt, eval_network=eval_net, metrics={"mae1": mae1, "mae2": mae2})
    multi_eval_dataset = create_multilabel_dataset(num_data=80)
    result = model.eval(multi_eval_dataset, dataset_sink_mode=False)
    print(result)
    ```

    The implementation results are as follows:

    ```text
    {'mae1': 8.572821712493896, 'mae2': 8.346409797668457}
    ```

    - When performing model evaluation, the output of the evaluation network will be transmitted to the `update` function of the evaluation metrics. In other words, the `update` function will receive three inputs, which are `logits`, `label1` and `label2`. `nn.MAE` only allows to calculate evaluation metrics on two inputs. Therefore, `set_indexes` is used to specify `mae1` to calculate evaluation results using inputs with subscripts 0 and 1, i.e. `logits` and `label1`. It is also used to specify `mae2` to calculate evaluation results using inputs with subscripts 0 and 2, i.e. `logits` and `label2`.

    - In practice, it is often necessary for all tags to participate in the evaluation. In this case, you need to customize `Metric` to flexibly use all outputs of the evaluation network to calculate the evaluation results. The details of the `Metric` customized method can be found at: <https://www.mindspore.cn/docs/programming_guide/en/r1.6/self_define_metric.html>.

6. Inference

   `Model` does not provide parameters for specifying the custom inference network, so you can run the forward network directly to get the inference results.

    ```python
    for d in multi_eval_dataset.create_dict_iterator():
        data = d["data"]
        break

    output = net(data)
    print(output)
    ```

     The implementation results are as follows:

    ```text
    [[ 7.147398  ]
    [ 3.4073524 ]
    [ 7.1618156 ]
    [ 1.8599509 ]
    [ 0.8132744 ]
    [ 4.92359   ]
    [ 0.6972816 ]
    [ 6.6525955 ]
    [ 1.2478441 ]
    [ 2.791972  ]
    [-1.2134678 ]
    [ 7.424588  ]
    [ 0.24634433]
    [ 7.15598   ]
    [ 0.68831706]
    [ 6.171982  ]]
    ```  

### Custom Training Network

When customizing `TrainOneStepCell`, you need to manually build the training network as `network` of `Model`, where `loss_fn` and `optimizer` both use the default value `None`. Then, `Model` will use `network` as the training network without any encapsulation.

Scenarios for customizing `TrainOneStepCell` can be found in [Build Training and Evaluating Network](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/train_and_eval.html). The following is a simple example, where `loss_net` and `opt` are `CustomWithLossCell` and `Momentum` as defined in the previous section.

```python
from mindspore.nn import TrainOneStepCell as CustomTrainOneStepCell
from mindspore import Model
from mindspore.train.callback import LossMonitor

# Build the Training Network Manually
train_net = CustomTrainOneStepCell(loss_net, opt)
# Define `Model` and Perform Training
model = Model(train_net)
multi_train_ds = create_multilabel_dataset(num_data=160)
model.train(epoch=1, train_dataset=multi_train_ds, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

The implementation results are as follows:

```text
epoch: 1 step: 1, loss is 8.834492
epoch: 1 step: 2, loss is 9.452023
epoch: 1 step: 3, loss is 6.974942
epoch: 1 step: 4, loss is 5.8168106
epoch: 1 step: 5, loss is 5.6446257
epoch: 1 step: 6, loss is 4.7653127
epoch: 1 step: 7, loss is 4.059086
epoch: 1 step: 8, loss is 3.5931993
epoch: 1 step: 9, loss is 2.8107128
epoch: 1 step: 10, loss is 2.3682175
```

Here `train_net` is the training network. When customizing the training network, you also need to customize the evaluating network. The way to perform model evaluation and inference is same as `Connect Forward Network with Loss Function Manually` in the previous section.

When both the label and the predicted value of the custom training network are single values, the evaluation function does not require special treatment, such as customization or using `set_indexes`. However, it is still necessary to pay attention to the correct usage of evaluation metrics in other scenarios.

### Weight Sharing of Custom Network

The weight sharing mechanism has been introduced in [Build Training and Evaluating Network](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/train_and_eval.html). When using MindSpore to build different network structures, as long as they are encapsulated in a same instance, all weights in this instance are shared. So, if there is any weight change in one network structure, the weights in other network structures will be changed simultaneously.

When using Model for training, for simple scenarios, `Model` internally uses `nn.WithLossCell`, `nn.TrainOneStepCell` and `nn.WithEvalCell` to build training and evaluating networks based on the forward `network` Instance. `Model` itself ensures weight sharing among inference, training, and evaluating networks. However, for custom scenarios, users need to be aware that the forward network should be instantiated only once. If the forward network is instantiated separately when building the training network and the evaluating network, you need to manually load the weights in the training network when using `eval` for model evaluation. Otherwise the model evaluation will use the initial weight values.
