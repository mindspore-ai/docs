# Callback Mechanism

`Ascend` `GPU` `CPU` `Model Development`

<!-- TOC -->

- [Callback Mechanism](#callback-mechanism)
    - [Overview](#overview)
    - [MindSpore Built-in Callback Functions](#mindspore-built-in-callback-functions)
    - [MindSpore Custom Callback Functions](#mindspore-custom-callback-functions)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/callback.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

The callback function is implemented as a class in MindSpore. The callback mechanism is similar to a monitoring mode, which helps you observe parameter changes and network internal status during network training. You can also perform operations based on specified conditions. During the training, the callback list executes the callback functions in the defined sequence. The callback mechanism enables you to effectively learn the training status of network models in time and make adjustments as required, greatly improving development efficiency.

In MindSpore, the callback mechanism is generally used in the network training process `model.train`. You can configure different built-in callback functions to transfer different parameters to implement various functions. For example, `LossMonitor` monitors the loss change of each epoch, `ModelCheckpoint` saves network parameters and models for re-training or inference, and `TimeMonitor` monitors the training time of each epoch and each step, terminates the training in advance, and dynamically adjusts parameters.

## MindSpore Built-in Callback Functions

- ModelCheckpoint

    This function is combined with the model training process, and saves the model and network parameters after training to facilitate re-inference or re-training. `ModelCheckpoint` is generally used together with `CheckpointConfig`. `CheckpointConfig` is a parameter configuration class that can be used to customize the checkpoint storage policy.

    For details, see [Saving Models](https://www.mindspore.cn/docs/programming_guide/en/master/save_model.html).

- SummaryCollector

    This function collects common information, such as loss, learning rate, computational graph, and parameter weight, helping you visualize the training process and view information. In addition, you can perform the summary operation to collect data from the summary file.

    For details, see [Collecting Summary Record](https://www.mindspore.cn/mindinsight/docs/en/master/summary_record.html).

- LossMonitor

    This function monitors the loss change during training. When the loss is NAN or INF, the training is terminated in advance. Loss information can be recorded in logs for you to view.

    For details, see the [Custom Debugging Information](https://www.mindspore.cn/docs/programming_guide/en/master/custom_debugging_info.html#mindsporecallback).

- TimeMonitor

    This function monitors the running time of each epoch and each step during training.

## MindSpore Custom Callback Functions

MindSpore provides powerful built-in callback functions and allows you to customize callback functions. For special requirements, you can customize callback functions based on the Callback base class. The callback function records important information during training and transfers the information to the callback object through a dictionary variable cb_params. You can obtain related attributes from each custom callback function and perform custom operations.

The following examples are used to introduce the custom callback functions:

1. Set a time threshold to terminate the training within a specified time. When the training time reaches the threshold, the training process is terminated.

2. Save the checkpoint file with the highest accuracy during training. You can customize the function to save a model with the highest accuracy after each epoch.

For details, see [Custom Callback](https://www.mindspore.cn/docs/programming_guide/en/master/custom_debugging_info.html#custom-callback).

According to the tutorial, you can easily customize other callback functions. For example, customize a function to output the detailed training information, including the training progress, training step, training name, and loss value, after each training is complete; terminate training when the loss or model accuracy reaches a certain value by setting the loss or model accuracy threshold. When the loss or model accuracy reaches the threshold, the training is terminated in advance.
