# Accuracy-Sensitive Detection

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/sdc.md)

## Overview

### Background

During model training, processors may encounter accuracy-sensitive anomalies, resulting in computational errors without reporting. Accuracy-sensitive anomalies may seriously affect model training.

### Solution

The MindSpore framework provides a solution for accuracy-sensitive detection of Transformer structure models.

For default feature value checkpoints, users can enable detection capability using the environment variable `NPU_ASD_ENABLE=1`, and adjust the detection intensity by configuring the environment variables `NPU_ASD_UPPER_THRESH` and `NPU_ASD_SIGMA_THRESH`.

In addition, the MindSpore framework supports users to customize feature value checkpoints according to their needs, further enhancing the detection capability for accuracy-sensitive anomalies.

For information on configuring related environment variables, see **Feature Switches and Configuration**.

For an introduction to default feature value checkpoints, and design guidelines for custom feature value checkpoints, see **Usage Recommendations and Detection Principles**.

### Usage Recommendations and Detection Principles

When processors encounter accuracy-sensitive anomalies, erroneous results are calculated. Due to the structure of Transformer models, these erroneous calculation results will propagate.

Based on experimental results, the following empirical conclusions are drawn:

* Not all accuracy-sensitive anomalies necessarily affect model convergence and performance. In fact, most accuracy-sensitive anomalies do not have observable effects on the model. See [reference](https://dl.acm.org/doi/abs/10.1145/3579371.3589105).
* Statistically, accuracy-sensitive anomalies during the backpropagation calculation process have a much greater impact than during the forward calculation process.
* In parallel training scenarios, calculation error results will propagate due to parallel computation.
* Setting too many checkpoints will affect model training performance.
* Based on experiments on the sensitivity of calculation errors, the MindSpore framework defaults to selecting the `Norm` activation value gradient in the backpropagation calculation process as the detection feature value, with performance loss less than 2% based on **Llama 2 - 7B** testing.

After enabling the detection switch, during the backpropagation phase of training Transformer structure models, abnormality is determined by collecting the activation value gradients of the Norm layer through modified specified network layers, calling the detection operator, and using an algorithm to determine if an anomaly exists. If an anomaly occurs, training is terminated, the NPU status on the device where the anomaly is detected is set to Warning, and a fault event is reported.

The reasons for feature value anomalies can be divided into two categories: hardware errors and software errors, which can be referred to in the **Fault Handling** section for further analysis.

### Usage Restrictions

Currently, this feature only supports Atlas A2 training series products, detects Transformer-like models, and detects accuracy anomalies that occur during training with the bfloat16 data type.

## Feature Switches and Configuration

The environment variable `NPU_ASD_ENABLE` serves as a feature switch, `export NPU_ASD_ENABLE=1` to enable this feature; if this environment variable is not configured or `export NPU_ASD_ENABLE=0`, this feature is disabled.

The environment variable `NPU_ASD_UPPER_THRESH` controls the absolute numerical threshold of detection, in the format of integer pairs, where the first element controls the first-level threshold of absolute numerical values, and the second element controls the second-level threshold of absolute numerical values; reducing the threshold can detect smaller fluctuations in abnormal data, increase the detection rate, and increasing the threshold is the opposite. In the default case where this environment variable is not configured, `NPU_ASD_UPPER_THRESH=1000000,10000`.

The environment variable `NPU_ASD_SIGMA_THRESH` controls the relative numerical threshold of detection, in the same format as the above, where the first element controls the first-level threshold of numerical changes, and the second element controls the second-level threshold of numerical changes; by default, `NPU_ASD_SIGMA_THRESH=100000,5000`.

## Use Cases

> This document describes the usage methods and use cases of accuracy-sensitive detection.

### Model and Dataset Preparation

To provide a complete experience, here we implement the usage case of accuracy-sensitive detection based on the MindSpore Transformers Llama2 network.

For the model and dataset preparation process, see [Llama 2](https://mindformers.readthedocs.io/zh-cn/latest/docs/model_cards/llama2.html).

If already prepared, you can skip this section directly.

### Default Detection Process Use Case

Under the `mindspore.ops.silent_check` module, `LayerNormASD` has been implemented as an operator integrated with ASD detection capabilities.

If the feature switch is enabled, in `mindspore.ops.__init__`, the above operator will automatically replace `mindspore.ops.LayerNorm`, providing default detection capabilities.

### Custom Detection Process Use Case

If custom feature value detection is required beyond the default detection scenario, in addition to enabling the feature switch `NPU_ASD_ENABLE`, you also need to implement custom operators that integrate ASD detection capabilities based on `ASDBase Jit Class`.

Here we use MindSpore Transformers Llama2 as an example to implement accuracy-sensitive detection of feature values in the Embedding layer.

#### Confirmation of Feature Value Checkpoints

Check the implementation of `llama.llama_layer.LlamaEmbedding`, where we choose to collect the gradient of the `Gather` operator during backpropagation as the detection feature value.

```python
class LlamaEmbedding(Cell):
    def construct(self, input_ids):
        """Forward of vocab embedding."""
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32, mstype.int64], self.cls_name)
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output
```

#### Implementation of ASD Detection Operator

For this use case, it is necessary to implement a custom Gather operator that integrates ASD detection capabilities.

Under `llama.llama_layer`, implement the corresponding operator for the collection point and refer to the following use case and API method comments of `ops.silent_check.ASDBase` for implementation.

```python
class GatherASD(ASDBase):
    def __init__(self, *args, **kwargs):
        super().__init__(P.Gather, *args, **kwargs)
        self.pre_val, self.min_val, self.max_val, self.cnt = self.generate_params()

    def __call__(self, input_params, input_indices, axis):
        if self.enable_check:
            input_params = self.check_op(
                input_params, self.pre_val, self.min_val, self.max_val, self.cnt, None)
            self.cnt += 1
        return self.op(input_params, input_indices, axis)
```

And replace the default Gather operator in the Embedding layer with the custom GatherASD operator.

```python
class LlamaEmbedding(Cell):
    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32, param_init='normal',
                 parallel_optimizer=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.gather = GatherASD()# Gather()
```

## Detection Results and Handling

### Abnormal Detection Results

When no numerical anomalies are detected, the training task runs without impact.

When numerical anomalies are detected, the training task fails and alerts are reported. To locate the faulty device, do one of the following:

* Search application logs for **ERROR** level error logs with the keyword "accuracy sensitivity feature abnormal";
* Monitor the NPU health status: if Health Status displays Warning, Error Code displays 80818C00, and Error Information displays node type=SoC, sensor type=Check Sensor, event state=check fail;
* Check the [Ascend Device Plugin](https://github.com/Ascend/ascend-device-plugin) events, report error code 80818C00, event type is fault event, and the fault level is minor.

### Fault Handling

Isolate the abnormal device, resume training with checkpoint recovery; meanwhile, on the abnormal device, use the Ascend-DMI tool to perform AICore ERROR stress diagnostics to detect whether there are faulty NPUs on the device. For details, see [ToolBox User Guide](https://www.hiascend.com/document/detail/zh/mindx-dl/2046/dluserguide/toolboxug/toolboxug_000002.html) in the "ascend-dmi tool usage > fault diagnosis" section.

If a faulty card is detected on the abnormal device, contact Huawei engineers for maintenance and replacement; if all NPUs on the abnormal device are normal, it is a software-related issue triggering feature value overflow, and it is recommended to check the processes and operators'es causes.