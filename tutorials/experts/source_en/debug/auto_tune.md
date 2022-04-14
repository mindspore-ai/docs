# AutoTune

`Ascend` `Model Optimization`

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/experts/source_en/debug/auto_tune.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

AutoTune is a tool that uses hardware resources and automatically tune the performance of TBE operators. Comparing with manually debugging the performance of operator, it takes less time and labor cost, and a model with better performance can be obtained. This document mainly introduces how to use the AutoTune tool to Online tune. The detail guidelines about the AutoTune framework, function description, and the fault handling can be got in [AutoTune Guides](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/31d1d888/about-this-document).

## TuneMode

The AutoTune tool includes `RL` and `GA` tuning modes. The`RL`tuning mode mainly supports`broadcast`,`reduce`, and`elewise`operators. The`GA`tuning mode mainly supports`cube`operators. The more information about the GA, RL, and the operators supported by the two tune mode can be got in [Tune Mode](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/41bb2c07) and [Operators](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/74e08a9c/operator-list).

## EnvironmentVariables

When using the AutoTune tool to tune the operators, some environment variables need to be configured (Required).

```shell
# Run package installation directory
LOCAL_ASCEND=/usr/local/Ascend
# Run package startup depends path
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin:${LOCAL_ASCEND}/fwkacllib/bin:$PATH
export PYTHONPATH=${LOCAL_ASCEND}/fwkacllib/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/opp

# Offline tuning environment variables
export ENABLE_TUNE_DUMP=True
```

Try to find the detailed description of environment variables, or other optional environment variables descriptions in [Environment Variable](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/3f0a50ba/environment-variable-configuration).

## EnablingTune

The AutoTune tool supports two tuning modes, `Online tune` and `Offline Tune`.

1. Online Tune

  Set `auto_tune_mode` in context to turn on Online tune. The value of `auto_tune_mode` should be in `["NO_TUNE", "RL", "GA", "RL,GA"]`.

  NO_TUNE: turn off tune.

  RL: turn on RL tune.

  GA: turn on GA tune.

  RL,GA: turn on GA and RL at the same time, the tool will select RL or GA automatically according to different types of operators which are used in the network.

  Example of online tuning:

  ```python
  import mindspore.context as context
  context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", auto_tune_mode="GA,RL")
  ....
  ```

  After setting the above context, you can start the tuning according to the normal execution of the training script. During the execution of the use case, no operation is required. The result of the model is the result after tuning.

2. Offline Tune

  The Offline Tune is using the dump data (The output description file, and the binary file of operators) of network model (Generate when training network) to tune the operators. The method of Offline Tune and related environment variables can be found in [Offline Tune](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/2fa72dd0) in `CANN` development tool guide, which is not described here.

## TuningResult

After the tuning starts, a file named `tune_result_{timestamp}_pidxxx.json` will be generated in the working directory to record the tuning process and tuning results. Please refer to [tuning result file analysis](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/b6ae7c6a) for specific analysis of this file.

After the tuning is complete. The custom knowledge base will be generated if the conditions are met. If the `TUNE_BANK_PATH`(Environment variable of the knowledge base storage path) is specified, the knowledge base(generated after tuning) will be saved in the specified directory. Otherwise, the knowledge base will be in the following default path. Please refer to [Custom knowledge base](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/b6ae7c6a) for the storage path.

## MergeKnowledgeBase

After operator tuning, the generated tuning knowledge base supports merging, which is convenient for re-executing, or the other models.(Only the same Ascend AI Processor can be merged). The more specific merging methods can be found in [merging knowledge base](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/c1a94cfc/repository-merging).

## Notice

Pay attention to the following points when using the AutoTune tool:

1. The AutoTune tool can only be used on `Ascend` platform.

2. Ensure that the available disk space in the home directory of the user who performs tuning in the operating environment is at least 20 GB.

3. The AutoTune tool depends on some third-party software, For example: `TensorFlow` and `pciutils`. Get more information about the [Depends](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/480d602c/environment-setup).

4. The AutoTune tool can not support all TBE operators, and can not guarantee the operator will have a performance benefit after tune (The operator has reached the best performance after multi-networks and multi-debugging manually).

5. After the tuning tool is turned on, it is obvious that the compilation time of the perception operator becomes longer.
