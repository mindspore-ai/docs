<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/source_en/advanced/model/metric.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

# Evaluation Metrics

When a training task is complete, an evaluation function (Metric) is often required to evaluate the quality of a model. Different training tasks usually require different metric functions. For example, for a binary classification problem, common evaluation metrics include precision, recall, and the like. For a multiclass classification task, macro and micro may be used for evaluation.

MindSpore provides evaluation functions for most common tasks, such as `Accuracy`、`Precision`、`MAE` and `MSE`. The evaluation functions provided by MindSpore cannot meet the requirements of all tasks. In most cases, you need to customize metrics for a specific task to evaluate the trained model.

The following describes how to customize metrics and how to use metrics in `mindspore.train.Model`.

> For details, see [Evaluation Metrics](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.train.html#evaluation-metrics).

## Customized Metrics

The customized metric function needs to inherit the `mindspore.train.Metric` parent class and re-implement the `clear`, `update`, and `eval` methods in the parent class.

- `clear`: initializes related internal parameters.
- `update`: receives network prediction output and labels, computes errors, and updates internal evaluation results after each step.
- `eval`: computes the final evaluation result after each epoch ends.

The mean absolute error (MAE) algorithm is shown in formula (1):

$$ MAE=\frac{1}{n}\sum_{i=1}^n\lvert ypred_i - y_i \rvert \tag{1}$$

The following uses the simple MAE algorithm as an example to describe the `clear`, `update`, and `eval` functions and their usage.

```python
import numpy as np
import mindspore as ms

class MyMAE(ms.train.Metric):
    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """Initialize variables _abs_error_sum and _samples_num."""
        self._abs_error_sum = 0  # Save error sum.
        self._samples_num = 0    # Accumulated data volume.

    def update(self, *inputs):
        """Update _abs_error_sum and _samples_num."""
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        # Compute the absolute error between the predicted value and the actual value.
        abs_error_sum = np.abs(y - y_pred)
        self._abs_error_sum += abs_error_sum.sum()

        # Total number of samples
        self._samples_num += y.shape[0]

    def eval(self):
        """Compute the final evaluation result.""
        return self._abs_error_sum / self._samples_num

# The network has two outputs.
y_pred = ms.Tensor(np.array([[0.1, 0.2, 0.6, 0.9], [0.1, 0.2, 0.6, 0.9]]), ms.float32)
y = ms.Tensor(np.array([[0.1, 0.25, 0.7, 0.9], [0.1, 0.25, 0.7, 0.9]]), ms.float32)

error = MyMAE()
error.clear()
error.update(y_pred, y)
result = error.eval()
print(result)
```

```text
    0.1499999612569809
```

## Using Metrics in Model Training

[mindspore.train.Model](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/train/mindspore.train.Model.html#mindspore.train.Model) is a high-level API used for training and evaluation. You can import customized or MindSpore existing metrics as parameters. Models can automatically call the imported metrics for evaluation.

After network model training, metrics need to be used to evaluate the training effect of the network model. Therefore, before specific code is demonstrated, you need to prepare a dataset, load the dataset, and define a simple linear regression network model.

$$f(x)=w*x+b \tag{2}$$

```python
import numpy as np
from mindspore import dataset as ds

def get_data(num, w=2.0, b=3.0):
    """Generate data and corresponding labels."""
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16):
    """Load the dataset."""
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset
```

### Using Built-in Evaluation Metrics

When the built-in metrics of MindSpore are transferred to `Model` as parameters, the metrics can be defined as a dictionary type. The `key` of the dictionary is a character string, and the `value` of the dictionary is the built-in [evaluation metric](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.train.html#evaluation-metrics) of MindSpore. The following example uses `train.Accuracy` to compute the classification accuracy.

```python
import mindspore.nn as nn
from mindspore.train import Model, MAE, LossMonitor

net = nn.Dense(1, 1)
loss_fn = nn.L1Loss()
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# Define a model and use the built-in Accuracy function.
model = Model(net, loss_fn, optimizer, metrics={"MAE": MAE()})

train_dataset = create_dataset(num_data=160)
eval_dataset = create_dataset(num_data=160)
train_dataset_size = train_dataset.get_dataset_size()

model.fit(10, train_dataset, eval_dataset, callbacks=LossMonitor(train_dataset_size))
```

```text
    epoch: 1 step: 10, loss is 5.908090114593506
    Eval result: epoch 1, metrics: {'MAE': 5.1329233884811405}
    epoch: 2 step: 10, loss is 3.9280264377593994
    Eval result: epoch 2, metrics: {'MAE': 3.0886757612228393}
    epoch: 3 step: 10, loss is 2.9104671478271484
    Eval result: epoch 3, metrics: {'MAE': 2.461756193637848}
    epoch: 4 step: 10, loss is 1.8725224733352661
    Eval result: epoch 4, metrics: {'MAE': 2.11311993598938}
    epoch: 5 step: 10, loss is 2.1637942790985107
    Eval result: epoch 5, metrics: {'MAE': 1.6749439239501953}
    epoch: 6 step: 10, loss is 1.3848766088485718
    Eval result: epoch 6, metrics: {'MAE': 1.317658966779709}
    epoch: 7 step: 10, loss is 1.052016258239746
    Eval result: epoch 7, metrics: {'MAE': 1.043285644054413}
    epoch: 8 step: 10, loss is 1.1781564950942993
    Eval result: epoch 8, metrics: {'MAE': 0.8706761479377747}
    epoch: 9 step: 10, loss is 0.8200418949127197
    Eval result: epoch 9, metrics: {'MAE': 0.7817940771579742}
    epoch: 10 step: 10, loss is 0.7065591812133789
    Eval result: epoch 10, metrics: {'MAE': 0.7885207533836365}
```

### Using Customized Evaluation Metrics

In the following example, the customized evaluation metric `MAE()` is transferred to `Model`, and the evaluation dataset is transferred to the `model.eval()` API for evaluation.

The validation result is of the dictionary type. The `key` of the validation result is the same as that of `metrics`. The `value` of the `metrics`result is the mean absolute error between the predicted value and the actual value.

```python
train_dataset = create_dataset(num_data=160)
eval_dataset = create_dataset(num_data=160)

model = Model(net, loss_fn, optimizer, metrics={"MAE": MyMAE()})

# Define a model and transfer the customized  metrics function MAE to the model.
model.fit(10, train_dataset, eval_dataset, callbacks=LossMonitor(train_dataset_size))
```

```text
    epoch: 1 step: 10, loss is 0.7992362380027771
    Eval result: epoch 1, metrics: {'MAE': 0.8640150725841522}
    epoch: 2 step: 10, loss is 0.8377518653869629
    Eval result: epoch 2, metrics: {'MAE': 0.9286439001560212}
    epoch: 3 step: 10, loss is 0.894376277923584
    Eval result: epoch 3, metrics: {'MAE': 0.8669328391551971}
    epoch: 4 step: 10, loss is 0.8098692893981934
    Eval result: epoch 4, metrics: {'MAE': 0.9018074989318847}
    epoch: 5 step: 10, loss is 0.8556416630744934
    Eval result: epoch 5, metrics: {'MAE': 0.8721640467643738}
    epoch: 6 step: 10, loss is 0.8508825302124023
    Eval result: epoch 6, metrics: {'MAE': 0.8601282179355622}
    epoch: 7 step: 10, loss is 0.7443522810935974
    Eval result: epoch 7, metrics: {'MAE': 0.9004024684429168}
    epoch: 8 step: 10, loss is 0.7394096851348877
    Eval result: epoch 8, metrics: {'MAE': 0.9380556881427765}
    epoch: 9 step: 10, loss is 0.7989674210548401
    Eval result: epoch 9, metrics: {'MAE': 0.8629323005676269}
    epoch: 10 step: 10, loss is 0.6581473350524902
    Eval result: epoch 10, metrics: {'MAE': 0.9144346475601196}
```
