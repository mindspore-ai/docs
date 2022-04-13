# Network Debugging

Translator: [Soleil](https://gitee.com/deng-zhihua)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/neural_network_debug.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

This chapter introduces the basic principles and common tools of Network Debugging, as well as some solutions to some common problems.

## The Basic Process of Network Debugging

The process of Network Debugging is divided into the following steps：

1. The network process is successfully debugged with no error during the whole process of network execution, proper output of loss value, and normal completion of parameter update.

   In general, if you use the `model.train` interface to execute a step completely without receiving an error, it means that it is executed normally and completed the parameter update; if you need to confirm accurately, you can save the checkpoint file for two consecutive steps by using the parameter `save_checkpoint_ steps=1` in `mindspore.train.callback.CheckpointConfig`, or use the `save_checkpoint` interface to save the Checkpoint file directly, and then print the weight values in the Checkpoint file with the following code to see if the weights of the two steps have changed. Finally, the update is completed.

   ```python
   import mindspore
   import numpy as np
   ckpt = mindspore.load_checkpoint(ckpt_path)
   for param in ckpt:
       value = ckpt[param].data.asnumpy()
       print(value)
   ```

2. Multiple iterations of the network are executed to output the loss values, and the loss values have a basic convergence trend.

3. Network accuracy debugging and hyper-parameter optimization.

## Common Methods Used in Network Debugging

### Process Debugging

This section introduces the problems and solutions during Network Debugging process after the script development is generally completed.

#### Process Debugging with PyNative Mode

For script development and network process debugging, we recommend using the PyNative mode for debugging. The PyNative mode supports executing single operators, normal functions and networks, as well as separate operations for computing gradients. In PyNative mode, you can easily set breakpoints and get intermediate results of network execution, and you can also debug the network by means of pdb.

By default, MindSpore is in Graph mode, which can be set as PyNative mode via `context.set_context(mode=context.PYNATIVE_MODE)`. Related examples can be found in [Debugging With PyNative Mode](hhttps://www.mindspore.cn/tutorials/zh-CN/master/advanced/pynative_graph/pynative.html).

#### Getting More Error Messages

During the network process debugging, if you need to get more information about error messages, you can get it by the following ways:

- Using pdb for debugging in PyNative mode, and using pdb to print relevant stack and contextual information to help locate problems.
- Using Print operator to print more contextual information. Related examples can be found in [Print Operator Features](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#print).
- Adjusting the log level to get more error information. MindSpore can easily adjust the log level through environment variables. Related examples can be found in [Logging-related Environment Variables And Configurations](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#id6).

#### Common Errors

During network process debugging, the common errors are as follows:

- The operator execution error.

  During the network process debugging, operator execution errors are often reported such as shape mismatch and unsupported dtype. Then, according to the error message, you should check whether the operator is used correctly and whether the shape of the input data is consistent with the expectation and make corresponding modifications.

  Supports for related operators and API introductions can be found in [Operator Support List](https://www.mindspore.cn/docs/en/master/note/operator_list.html) and [Operators Python API](https://www.mindspore.cn/docs/en/master/index.html).

- The same script works in PyNative mode, but reports bugs in Graph mode.

  In MindSpore's Graph mode, the code in the `construct` function is parsed by the MindSpore framework, and there is some Python syntax that is not yet supported which results in errors. In this case, you should follow [MindSpore's Syntax Description](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) according to the error message.

- Distributed parallel training script is misconfigured.

  Distributed parallel training scripts and environment configuration can be found in [Distributed Parallel Training Tutorial](https://www.mindspore.cn/docs/en/master/design/distributed_training_design.html).

### Loss Value Comparison

With a benchmark script, the loss values run by the benchmark script can be compared with those run by the MindSpore script, which can be used to verify the correctness of the overall network structure and the accuracy of the operator.

#### Main Steps

1. Guaranteeting Identical Input

    It is necessary to ensure that the inputs are the same in both networks, so that they can have the same network output in the same network structure. The same inputs can be guaranteed using following ways:

    - Using numpy to construct the input data to ensure the same inputs to the networks. MindSpore supports free conversion of Tensor and numpy. The following script can be used to construct the input data.

      ```python
      input = Tensor(np.random.randint(0, 10, size=(3, 5, 10)).astype(np.float32))
      ```

    - Using the same dataset for computation. MindSpore supports the use of the TFRecord dataset, which can be read by using the `mindspore.dataset.TFRecordDataset` interface.

2. Removing the Influence of Randomness in the Network

   The main methods to remove the effect of randomness in the network are to set the same randomness seed, turn off the data shuffle, modify the code to remove the effect of random operators in the network such as dropout and initializer, etc.

3. Ensuring the Same Settings for the Relevant Hyperparameters

   It is necessary to ensure the same settings for the hyperparameters in the network in order to guarantee the same input and the same output of the operator.

4. Running the network and comparing the output loss values. Generally, the error of the loss value is about 1‰. Because the operator itself has a certain accuracy error. As the number of steps increases, the error will have a certain accumulation.

#### Related Issues Locating

If the loss errors are large, the problem locating can be done by using following ways:

- Checking whether the input and hyperparameter settings are the same, and whether the randomness effect is completely removed.

  if the loss value differs significantly after multiple executions of the same script, it means that the effect of randomness in the network is not completely removed.

- Overall judgment.

  If there is a large error in the first iteration of loss values, it means that there is a problem with the forward calculation of the network.

  If the loss value in the first iteration is within the error range but the second iteration starts with a large error in the loss value, it means that there should be no problem in the forward calculation of the network and there may be problems in the reverse gradient and weight update calculation.

- After having the overall judgment, compare the accuracy of input and output values from rough to fine.

  First, compare the input and output values layer by layer for each subnet starting from the input, and identify the subnets that initially have problems.

  Then, compare the network structure in the subnet and the input and output of the operator, find the network structure or operator that occurs problems, and modify it.

  If you find any operator accuracy problems during the process, you can raise an issue on the [MindSpore Code Hosting Platform](https://gitee.com/mindspore/mindspore), and the relevant personnel will follow up on the problem.

- MindSpore provides various tools for acquiring intermediate network data, which can be used according to the actual situation.

    - [Data Dump function](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#dump)
    - [Use Print Operator To Print Related Information](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#print)
    - [Using The Visualization Component MindInsight](https://www.mindspore.cn/mindinsight/docs/en/master/index.html)

### Precision Debugging Tools

#### Customized Debugging Information

- [Callback Function](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#callback)

   MindSpore has provided ModelCheckpoint, LossMonitor, SummaryCollector and other Callback classes for saving model parameters, monitoring loss values, saving training process information, etc. Users can also customize Callback functions to implement starting and ending runs at each epoch and step, and please refer to [Custom Callback](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#id3) for specific examples.

- [MindSpore Metrics Function](https://www.mindspore.cn/tutorials/experts/en/master/debug/custom_debug.html#mindspore-metrics)

   When the training is finished, metrics can be used to evaluate the training results. MindSpore provides various metrics for evaluation, such as: `accuracy`, `loss`, `precision`, `recall`, `F1`, etc.

- Customized Learning Rate

   MindSpore provides some common implementations of dynamic learning rate and some common optimizers with adaptive learning rate adjustment functions, and [Dynamic Learning Rate](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#dynamic-learning-rate) and [Optimizer Functions](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#optimizer-functions) in the API documentation can be found.

   At the same time, the user can implement a customized dynamic learning rate, as exemplified by WarmUpLR:

   ```python
   class WarmUpLR(LearningRateSchedule):
       def __init__(self, learning_rate, warmup_steps):
           super(WarmUpLR, self).__init__()
           ## check the input
           if not isinstance(learning_rate, float):
               raise TypeError("learning_rate must be float.")
           validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
           validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)
           ## define the operators
           self.warmup_steps = warmup_steps
           self.learning_rate = learning_rate
           self.min = ops.Minimum()
           self.cast = ops.Cast()

       def construct(self, global_step):
           ## calculate the lr
           warmup_percent = self.cast(self.min(global_step, self.warmup_steps), mstype.float32)/ self.warmup_steps
           return self.learning_rate * warmup_percent
   ```

#### Hyper-Parameter Optimization with MindOptimizer

MindSpore provides MindOptimizer tools to help users perform hyper-parameter optimization conveniently, and the detailed examples and usage methods can be found in [Hyper-Parameter Optimization With MindOptimizer](https://www.mindspore.cn/mindinsight/docs/en/master/hyper_parameters_auto_tuning.html).

#### Loss Value Anomaly Locating

For cases where the loss value is INF, NAN, or the loss value does not converge, you can investigate the following scenarios:

1. Checking for loss_scale overflow.

   In the scenario of using loss_scale with mixed precision, the situation that the loss value is INF and NAN may be caused by the scale value being too large. If it is dynamic loss_scale, the scale value will be adjusted automatically; if it is static loss_scale, the scale value needs to be reduced.

   If the `scale=1` case still has a loss value of INF, NAN, there should be an overflow of operators in the network and further investigation for locating the problem is needed.

2. The causes of abnormal loss values may be caused by abnormal input data, operator overflow, gradient disappearance, gradient explosion, etc.

   To check the intermediate value of the network such as operator overflow, gradient of 0, abnormal weight, gradient disappearance and gradient explosion, it is recommended to use [MindInsight Debugger](https://www.mindspore.cn/mindinsight/docs/en/master/debugger.html) to set the corresponding detection points for detection and debugging, which can locate the problem in a more comprehensive way with the strong debuggability.

   The following are a few simple initial troubleshooting methods:

    - Observing whether the weight values appear or loading the saved Checkpoint file to print the weight values can make a preliminary judgment. Printing the weight values can refer to the following code:

        ```python
        import mindspore
        import numpy as np
        ckpt = mindspore.load_checkpoint(ckpt_path)
        for param in ckpt:
            value = ckpt[param].data.asnumpy()
            print(value)
        ```

    - Checking whether the gradient is 0 or comparing whether the weight values of Checkpoint files saved in different steps have changed can make a preliminary judgment. The comparison of the weight values of Checkpoint files can be referred to the following code:

        ```python
        import mindspore
        import numpy as np
        ckpt1 = mindspore.load_checkpoint(ckpt1_path)
        ckpt2 = mindspore.load_checkpoint(ckpt2_path)
        sum = 0
        same = 0
        for param1,param2 in zip(ckpt1,ckpt2):
            sum = sum + 1
            value1 = ckpt[param1].data.asnumpy()
            value2 = ckpt[param2].data.asnumpy()
            if value1 == value2:
                print('same value: ', param1, value1)
                same = same + 1
        print('All params num: ', sum)
        print('same params num: ', same)
        ```

    - Checking whether there is NAN, INF abnormal data in the weight value, you can also load the Checkpoint file for a brief judgment. In general, if there is NAN, INF in the weight value, there is also NAN, INF in the gradient calculation, and there may be an overflow situation. The relevant code reference is as follows:

        ```python
        import mindspore
        import numpy as np
        ckpt = mindspore.load_checkpoint(ckpt_path)
        for param in ckpt:
            value = ckpt[param].data.asnumpy()
            if np.isnan(value):
                print('NAN value:', value)
            if np.isinf(value):
                print('INF value:', value)
        ```
