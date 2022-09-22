# Evaluation Metrics

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/model/metric.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

When a training task is complete, an evaluation function (Metric) is often required to evaluate the quality of a model. Different training tasks usually require different metric functions. For example, for a binary classification problem, common evaluation metrics include precision, recall, and the like. For a multiclass classification task, macro and micro may be used for evaluation.

MindSpore provides evaluation functions for most common tasks, such as `nn.Accuracy`, `nn.Precision`, `nn.MAE`, and `nn.MSE`. The evaluation functions provided by MindSpore cannot meet the requirements of all tasks. In most cases, you need to customize metrics for a specific task to evaluate the trained model.

The following describes how to customize metrics and how to use metrics in `nn.Model`.

> For details, see [Evaluation Metrics](https://www.mindspore.cn/docs/en/master/api_python/mindspore.train.html#evaluation-metrics).

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
import mindspore.nn as nn

class MyMAE(nn.Metric):
    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """Initialize variables _abs_error_sum and _samples_num."""
        self._abs_error_sum = 0  # Save error sum.
        self._samples_num = 0    # Accumulated data volume.

    @nn.rearrange_inputs
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
print("output(y_pred, y):", result)
```

```text
    output(y_pred, y): 0.1499999612569809
```

Note that if the network has multiple outputs in `update`, but only two outputs are used for evaluation, you can use the `set_indexes` method to rearrange the input of `update` to compute evaluation metrics.

To use the `set_indexes` method, you need to use the modifier `nn.rearrange_inputs` to modify the `update` method. Otherwise, the input configured using `set_indexes` does not take effect.

```python
# The network has three outputs: y_pred, y, and z.
z = ms.Tensor(np.array([[0.1, 0.25, 0.7, 0.8], [0.1, 0.25, 0.7, 0.8]]), ms.float32)

# Use y_pred and z for evaluation.
error = MyMAE().set_indexes([0, 2])
error.clear()
error.update(y_pred, y, z)
result = error.eval()
print("output(y_pred,z):", result)
```

```text
    output(y_pred,z): 0.24999992549419403
```

## Using Metrics in Model Training

[mindspore.train.Model](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model) is a high-level API used for training and evaluation. You can import customized or MindSpore existing metrics as parameters. Models can automatically call the imported metrics for evaluation.

After network model training, metrics need to be used to evaluate the training effect of the network model. Therefore, before specific code is demonstrated, you need to prepare a dataset, load the dataset, and define a simple linear regression network model.

$$f(x)=w*x+b \tag{2}$$

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import dataset as ds
from mindspore.common.initializer import Normal

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

class LinearNet(nn.Cell):
    """Define the linear regression network.""
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

loss = nn.L1Loss()
```

### Using Built-in Evaluation Metrics

When the built-in metrics of MindSpore are transferred to `Model` as parameters, the metrics can be defined as a dictionary type. The `key` of the dictionary is a character string, and the `value` of the dictionary is the built-in [evaluation metric](https://www.mindspore.cn/docs/en/master/api_python/mindspore.train.html#evaluation-metrics) of MindSpore. The following example uses `nn.Accuracy` to compute the classification accuracy.

```python
import mindspore.nn as nn
import mindspore as ms
from mindvision.engine.callback import LossMonitor

ds_train = create_dataset(num_data=160)
net = LinearNet()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# Define a model and use the built-in Accuracy function.
model = ms.Model(net, loss, opt, metrics={"MAE": nn.MAE()})
model.train(epoch=1, train_dataset=ds_train, callbacks=LossMonitor(0.005))

# Evaluate the model.
ds_eval = create_dataset(num_data=160)
output = model.eval(ds_eval)
print(output)
```

```text
    Epoch:[  0/  1], step:[    1/   10], loss:[10.206/10.206], time:148.661 ms, lr:0.00500
    Epoch:[  0/  1], step:[    2/   10], loss:[8.827/9.516], time:0.671 ms, lr:0.00500
    Epoch:[  0/  1], step:[    3/   10], loss:[13.232/10.755], time:0.681 ms, lr:0.00500
    Epoch:[  0/  1], step:[    4/   10], loss:[10.893/10.789], time:0.704 ms, lr:0.00500
    Epoch:[  0/  1], step:[    5/   10], loss:[8.339/10.299], time:0.668 ms, lr:0.00500
    Epoch:[  0/  1], step:[    6/   10], loss:[8.881/10.063], time:0.826 ms, lr:0.00500
    Epoch:[  0/  1], step:[    7/   10], loss:[6.288/9.524], time:0.923 ms, lr:0.00500
    Epoch:[  0/  1], step:[    8/   10], loss:[8.166/9.354], time:0.932 ms, lr:0.00500
    Epoch:[  0/  1], step:[    9/   10], loss:[7.538/9.152], time:0.932 ms, lr:0.00500
    Epoch:[  0/  1], step:[   10/   10], loss:[5.517/8.789], time:0.980 ms, lr:0.00500
    Epoch time: 167.900 ms, per step time: 16.790 ms, avg loss: 8.789
    {'MAE': 5.931522464752197}
```

### Using Customized Evaluation Metrics

In the following example, the customized evaluation metric `MAE()` is transferred to `Model`, and the evaluation dataset is transferred to the `model.eval()` API for evaluation.

The validation result is of the dictionary type. The `key` of the validation result is the same as that of `metrics`. The `value` of the `metrics`result is the mean absolute error between the predicted value and the actual value.

```python
ds_train = create_dataset(num_data=160)
net1 = LinearNet()
opt = nn.Momentum(net1.trainable_params(), learning_rate=0.005, momentum=0.9)

# Define a model and transfer the customized  metrics function MAE to the model.
model1 = ms.Model(net1, loss, opt, metrics={"MAE": MyMAE()})
model1.train(epoch=1, train_dataset=ds_train, callbacks=LossMonitor(0.005))

# Evaluate the model.
ds_eval = create_dataset(num_data=160)
output = model1.eval(ds_eval)
print(output)
```

```text
    Epoch:[  0/  1], step:[    1/   10], loss:[9.931/9.931], time:157.518 ms, lr:0.00500
    Epoch:[  0/  1], step:[    2/   10], loss:[10.705/10.318], time:0.751 ms, lr:0.00500
    Epoch:[  0/  1], step:[    3/   10], loss:[11.313/10.650], time:0.722 ms, lr:0.00500
    Epoch:[  0/  1], step:[    4/   10], loss:[9.445/10.349], time:0.738 ms, lr:0.00500
    Epoch:[  0/  1], step:[    5/   10], loss:[5.492/9.377], time:0.737 ms, lr:0.00500
    Epoch:[  0/  1], step:[    6/   10], loss:[8.060/9.158], time:0.839 ms, lr:0.00500
    Epoch:[  0/  1], step:[    7/   10], loss:[7.866/8.973], time:0.900 ms, lr:0.00500
    Epoch:[  0/  1], step:[    8/   10], loss:[7.264/8.760], time:0.863 ms, lr:0.00500
    Epoch:[  0/  1], step:[    9/   10], loss:[8.975/8.784], time:0.885 ms, lr:0.00500
    Epoch:[  0/  1], step:[   10/   10], loss:[7.630/8.668], time:0.958 ms, lr:0.00500
    Epoch time: 177.346 ms, per step time: 17.735 ms, avg loss: 8.668
    {'MAE': 5.533915233612061}
```
