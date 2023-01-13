# Evaluation Metrics

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/advanced/model/metric.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

When a training task is complete, an evaluation function (Metric) is often required to evaluate the quality of a model. Different training tasks usually require different metric functions. For example, for a binary classification problem, common evaluation metrics include precision, recall, and the like. For a multiclass classification task, macro and micro may be used for evaluation.

MindSpore provides evaluation functions for most common tasks, such as `nn.Accuracy`, `nn.Precision`, `nn.MAE`, and `nn.MSE`. The evaluation functions provided by MindSpore cannot meet the requirements of all tasks. In most cases, you need to customize metrics for a specific task to evaluate the trained model.

The following describes how to customize metrics and how to use metrics in `nn.Model`.

> For details, see [Evaluation Metrics](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.nn.html#evaluation-metrics).

## Customized Metrics

The customized metric function needs to inherit the `nn.Metric` parent class and re-implement the `clear`, `update`, and `eval` methods in the parent class.

- `clear`: initializes related internal parameters.
- `update`: receives network prediction output and labels, computes errors, and updates internal evaluation results after each step.
- `eval`: computes the final evaluation result after each epoch ends.

The mean absolute error (MAE) algorithm is shown in formula (1):

$$ MAE=\frac{1}{n}\sum_{i=1}^n\lvert ypred_i - y_i \rvert \tag{1}$$

The following uses the simple MAE algorithm as an example to describe the `clear`, `update`, and `eval` functions and their usage.

```python
import numpy as np
import mindspore as ms
from mindspore import nn

class MyMAE(nn.Metric):
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
        """Compute the final evaluation result."""
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

[mindspore.Model](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.Model.html#mindspore.Model) is a high-level API used for training and evaluation. You can import customized or MindSpore existing metrics as parameters. Models can automatically call the imported metrics for evaluation.

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

When the built-in metrics of MindSpore are transferred to `Model` as parameters, the metrics can be defined as a dictionary type. The `key` of the dictionary is a character string, and the `value` of the dictionary is the built-in [evaluation metric](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.nn.html#evaluation-metrics) of MindSpore. The following example uses `nn.Accuracy` to compute the classification accuracy.

```python
import mindspore.nn as nn
from mindspore.nn import MAE
from mindspore import Model, LossMonitor

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
epoch: 1 step: 10, loss is 6.0811052322387695
Eval result: epoch 1, metrics: {'MAE': 5.012505912780762}
epoch: 2 step: 10, loss is 2.7896716594696045
Eval result: epoch 2, metrics: {'MAE': 3.380072832107544}
epoch: 3 step: 10, loss is 3.0297815799713135
Eval result: epoch 3, metrics: {'MAE': 2.5002413272857664}
epoch: 4 step: 10, loss is 2.3680481910705566
Eval result: epoch 4, metrics: {'MAE': 2.4334578275680543}
epoch: 5 step: 10, loss is 1.8126990795135498
Eval result: epoch 5, metrics: {'MAE': 1.8317200541496277}
epoch: 6 step: 10, loss is 1.6006351709365845
Eval result: epoch 6, metrics: {'MAE': 1.521335732936859}
epoch: 7 step: 10, loss is 1.1064929962158203
Eval result: epoch 7, metrics: {'MAE': 1.2528185725212098}
epoch: 8 step: 10, loss is 0.9595810174942017
Eval result: epoch 8, metrics: {'MAE': 1.0719563841819764}
epoch: 9 step: 10, loss is 0.6517931222915649
Eval result: epoch 9, metrics: {'MAE': 0.9766222715377808}
epoch: 10 step: 10, loss is 0.9312882423400879
Eval result: epoch 10, metrics: {'MAE': 0.9238077104091644}
```

### Using Customized Evaluation Metrics

In the following example, the customized evaluation metric `MAE()` is transferred to `Model`, and the evaluation dataset is transferred to the `model.eval()` API for evaluation.

The validation result is of the dictionary type. The `key` of the validation result is the same as that of `metrics`. The `value` of the `metrics`result is the mean absolute error between the predicted value and the actual value.

```python
train_dataset = create_dataset(num_data=160)
eval_dataset = create_dataset(num_data=160)

model = Model(net, loss_fn, optimizer, metrics={"MAE": MyMAE()})

# Define a model and transfer the customized metrics function MAE to the model.
model.fit(10, train_dataset, eval_dataset, callbacks=LossMonitor(train_dataset_size))
```

```text
epoch: 1 step: 10, loss is 0.5679571628570557
Eval result: epoch 1, metrics: {'MAE': 0.7907268464565277}
epoch: 2 step: 10, loss is 0.8198273181915283
Eval result: epoch 2, metrics: {'MAE': 0.7729107916355134}
epoch: 3 step: 10, loss is 0.5721814036369324
Eval result: epoch 3, metrics: {'MAE': 0.7661101937294006}
epoch: 4 step: 10, loss is 0.6523740291595459
Eval result: epoch 4, metrics: {'MAE': 0.7704753875732422}
epoch: 5 step: 10, loss is 0.5641313791275024
Eval result: epoch 5, metrics: {'MAE': 0.7609358102083206}
epoch: 6 step: 10, loss is 0.774018406867981
Eval result: epoch 6, metrics: {'MAE': 0.7739883124828338}
epoch: 7 step: 10, loss is 0.7306548357009888
Eval result: epoch 7, metrics: {'MAE': 0.7757290184497834}
epoch: 8 step: 10, loss is 0.667199969291687
Eval result: epoch 8, metrics: {'MAE': 0.7627188444137574}
epoch: 9 step: 10, loss is 0.689708948135376
Eval result: epoch 9, metrics: {'MAE': 0.7673796474933624}
epoch: 10 step: 10, loss is 0.7661054134368896
Eval result: epoch 10, metrics: {'MAE': 0.7654164433479309}
```
