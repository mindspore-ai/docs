# Enabling AutoTune

`Linux` `Ascend` `AutoTune` `Operator Compile` `Intermediate` `Expert`

<!-- TOC -->

- [Enabling AutoTune](#Enabling-AutoTune)
    - [Overview](#Overview)
    - [TuneMode](#TuneMode)
    - [EnvironmentVariables](#EnvironmentVariables)
    - [EnablingTune](#EnablingTune)
    - [Notice](#Notice)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/enable_auto_tune.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## Overview

AutoTune is a tool that uses hardware resources and automatically tune the performance of TBE operators. Comparing with manually debugging the performance of operator, it takes less time and labor cost, and a model with better performance can be obtained. This document mainly introduces how to use the AutoTune tool to Online tune. The detail guidelines about the AutoTune framework, function description, and the fault handling can be got in [AutoTune Guides](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/31d1d888/about-this-document).

## TuneMode

The AutoTune tool includes `RL` and `GA` tuning modes. The`RL`tuning mode mainly supports`broadcast`,`reduce`, and`elewise`operators. The`GA`tuning mode mainly supports`cube`operators. The more information about the GA, RL, and the operators supported by the two tune mode can be got in [Tune Mode](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/41bb2c07) and [Operators](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/74e08a9c/operator-list).

## EnvironmentVariables

When using the AutoTune tool to tune the operators, some environment variables need to be configured (Required). Try to find the descriptions in [Environment Variable](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/3f0a50ba/environment-variable-configuration).

## EnablingTune

AutoTune tools support `Online` and `Offline` tuning mode. The Online tune can be turned on by set `auto_tune_mode` in context. The Offline is using the dump data (The output description file, and the binary file of operators) of network model (Generate when training network) to tune the operators. This document mainly introduces how to use online tuning, the usage of Offline tune can be got in [Offline Tune](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/2fa72dd0).

Set `auto_tune_mode` in context to turn on Online tune. The value of `auto_tune_mode` should be in `["NO_TUNE", "RL", "GA", "RL,GA"]`.

NO_TUNE: turn off tune.

RL: turn on RL tune.

GA: turn on GA tune.

RL,GA: turn on GA and RL at the same time, the tool will select RL or GA automatically according to different types of operators which used in the network.

Example of online tuning:

```python
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", auto_tune_mode="GA,RL")

......
```

## Notice

Pay attention to the following points when using the AutoTune tool:

1. The AutoTune tool can only be used on `Ascend` platform.

2. Ensure that the available disk space in the home directory of the user who performs tuning in the operating environment is at least 20 GB.

3. The AutoTune tool depends on some third-party software, For example: `TensorFlow` and `pciutils`. Get more information about the [Depends](https://support.huawei.com/enterprise/en/doc/EDOC1100206689/480d602c/environment-setup).

4. The AutoTune tool can not support all TBE operators, and can not guarantee the operator will have a performance benefit after tune (The operator has reached the best performance after multi-networks and multi-debugging manually).

5. After the tuning tool is turned on, it is obvious that the compilation time of the perception operator becomes longer.
