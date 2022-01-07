# Customize Metrics to Verify Model Evaluation Accuracy

`Ascend` `GPU` `CPU` `Model Evaluation`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/self_define_metric.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>
&nbsp;&nbsp;

## Overview

After training, it is necessary to use Metrics for model evaluation. Different tasks usually needs different Metrics. For example, precision and recall are often employed as the Metrics for binary classification task, and for multi-class classification tasks, we can use Metrics like Macro or Micro. MindSpore has provided many different Metrics such as `nn.Accuracy`,`nn.Pecision`,`nn.MAE`, `nn.Topk`, `nn.MSE`, etc. For more information, one can refer to [Metrics](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.nn.html#metrics). Although MindSpore has provided Mstrics for most common tasks, it cannot satisfy all need. Therefore, for some specific tasks, one can define his own Metrics to evaluate the model.

Next, we will introduce how to customize Metrics and use it in `nn.Model`.

## Steps of Customizing Metrics

All the Metrics need to inherite from `nn.Metric` class. Users only need to re-implement `clear`,`update` and `eval` methods. Specifically, `clear` is used to initialize internal parameter, `updata` receives network prediction and groundtruth to update the internal evaluation result. Finally, `eval` will calculate the metric and return the results. Next, we will take `MAE` for example to explain the 3 steps.

The sample code is as follows:

```python
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.nn import Metric
from mindspore.nn import rearrange_inputs

class MyMAE(Metric):

    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._abs_error_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += abs_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num
```

- `clear` method initializes two variables where `_abs_error_sum` is used to keep sum of errors and `_samples_num` is used to accumulate the number of data items.
- `update` method receives network prediction and groundtruth to calculate error and undate `_abs_error_sum` and `_samples_num`. Here, `_convert_data` converts `Tensor` to numpy, therefore, the following calculation can be implemented using numpy. If the output number of the evaluation model is more than 2, users can use decorator `rearrange_inputs` to specify which two outputs should be used for error calculation. Usually, the `update` method is employed every step to calculate and update some internal values.
- `eval` methods calculate the final metric and return it. Usually, the `eval` method is employed each step to calculate final result.

Next, we try to calculate the MAE result of several data items using the code above.

```python
x = Tensor(np.array([[0.1, 0.2, 0.6, 0.9],[0.1, 0.2, 0.6, 0.9]]), mindspore.float32)
y = Tensor(np.array([[0.1, 0.25, 0.7, 0.9],[0.1, 0.25, 0.7, 0.9]]), mindspore.float32)
error = MyMAE()
error.clear()
error.update(x, y)
result = error.eval()
print(result)

```

```python
0.1499919

```

```python
x = Tensor(np.array([[0.1, 0.2, 0.6, 0.9],[0.1, 0.2, 0.6, 0.9]]), mindspore.float32)
y = Tensor(np.array([[0.1, 0.25, 0.7, 0.9],[0.1, 0.25, 0.7, 0.9]]), mindspore.float32)
z = Tensor(np.array([[0.1, 0.25, 0.7, 0.9],[0.1, 0.25, 0.7, 0.9]]), mindspore.float32)
error = MyMAE().set_indexes([0, 2])
error.clear()
error.update(x, y, z)
result = error.eval()
print(result)

```

```python
0.1499919

```

## Using Metrics in Model

`mindspore.Model` is high-level API for training or testing. The custom Metrics can be passed to Model, Model will automatically call it to evaluate the trained model. The example code is as follows. More information about `mindspore.Model`, please refer to [Model](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.Model.html#mindspore.Model).

```python
model = Model(network, loss, metrics={"MyMAE":MyMAE()})
output = model.eval(eval_dataset)
```


