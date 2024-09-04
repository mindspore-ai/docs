# Feature Value Detection

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/debug/sdc.md)

## Overview

### Background

During model training, processors may encounter feature value detection anomalies, resulting in computational errors without reporting. Feature value detection anomalies may seriously affect model training.

### Solution

The MindSpore framework provides a solution for feature value detection of Transformer structure models.

For default feature value detection checkpoints, users can enable detection capability using the environment variable `NPU_ASD_ENABLE=1`, `NPU_ASD_ENABLE=2`  or `NPU_ASD_ENABLE=3`, and adjust the detection intensity by configuring the environment variables `NPU_ASD_UPPER_THRESH` and `NPU_ASD_SIGMA_THRESH`.

In addition, the MindSpore framework supports users to customize feature value detection checkpoints according to their needs, further enhancing the detection capability for feature value detection anomalies.

For information on configuring related environment variables, see **Feature Switches and Configuration**.

For an introduction to default feature value detection checkpoints, and design guidelines for custom feature value detection checkpoints, see **Usage Recommendations and Detection Principles**.

### Usage Recommendations and Detection Principles

When processors encounter feature value detection anomalies, erroneous results are calculated. Due to the structure of Transformer models, these erroneous calculation results will propagate.

Based on experimental results, the following empirical conclusions are drawn:

* Not all feature value detection anomalies necessarily affect model convergence and performance. In fact, most feature value detection anomalies do not have observable effects on the model. See [reference](https://dl.acm.org/doi/abs/10.1145/3579371.3589105).
* Statistically, feature value detection anomalies during the backpropagation calculation process have a much greater impact than during the forward calculation process.
* In parallel training scenarios, calculation error results will propagate due to parallel computation.
* Setting too many checkpoints will affect model training performance.
* Based on experiments on the sensitivity of calculation errors, the MindSpore framework defaults to selecting the `Norm` activation value gradient in the backpropagation calculation process as the detection feature value, with performance loss less than 2% based on **Llama 2 - 7B** testing.

After enabling the detection switch, during the backpropagation phase of training Transformer structure models, abnormality is determined by collecting the activation value gradients of the Norm layer through modified specified network layers, calling the detection operator, and using an algorithm to determine if an anomaly exists. If an anomaly occurs, training is terminated, the NPU status on the device where the anomaly is detected is set to Warning, and a fault event is reported.

The reasons for feature value anomalies can be divided into two categories: hardware errors and software errors, which can be referred to in the **Fault Handling** section for further analysis.

### Usage Restrictions

Currently, this feature only supports Atlas A2 training series products, detects abnormal feature value during the training process with Transformer model and bfloat16 data type.

## Feature Switches and Configuration

The environment variable `NPU_ASD_ENABLE` serves as a feature switch, `export NPU_ASD_ENABLE=1`, `export NPU_ASD_ENABLE=2` or `export NPU_ASD_ENABLE=3` to enable this feature; if this environment variable is not configured or `export NPU_ASD_ENABLE=0`, this feature is disabled.

The environment variable `NPU_ASD_UPPER_THRESH` controls the absolute numerical threshold of detection, in the format of integer pairs, where the first element controls the first-level threshold of absolute numerical values, and the second element controls the second-level threshold of absolute numerical values; reducing the threshold can detect smaller fluctuations in abnormal data, increase the detection rate, and increasing the threshold is the opposite. In the default case where this environment variable is not configured, `NPU_ASD_UPPER_THRESH=1000000,10000`.

The environment variable `NPU_ASD_SIGMA_THRESH` controls the relative numerical threshold of detection, in the same format as the above, where the first element controls the first-level threshold of numerical changes, and the second element controls the second-level threshold of numerical changes; by default, `NPU_ASD_SIGMA_THRESH=100000,5000`.

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