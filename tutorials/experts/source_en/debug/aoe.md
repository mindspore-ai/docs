# Ascend Optimization Engine (AOE)

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/aoe.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

Ascend Optimization Engine (AOE) is an automatic tuning tool that makes full use of limited hardware resources to meet the performance requirements of operators and the entire network. The AOE tool includes online and offline tuning modes. The more information about the online, offline, and the operators supported by the two tune mode can be got in [Tune Mode](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoe_16_001.html). This document mainly introduces how to use the AOE to tune in MindSpore training scenarios.

## Enabling Tune

1. Online Tune

  Set `aoe_tune_mode` in set_context to turn on Online tune. The value of `auto_tune_mode` should be `online`.

  Example of online tuning:

  ```python
  import mindspore as ms
  ms.set_context(aoe_tune_mode="online")
  ....
  ```

  After setting the above context, you can start the tuning according to the normal execution of the training script. During the execution of the use case, no operation is required. The result of the model is the result after tuning.

2. Offline Tune

  The Offline Tune is using the dump data (The output description file, and the binary file of operators) of network model (Generate when training network) to tune the operators. The method of Offline Tune and related environment variables can be found in [Offline Tune](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoe_16_023.html) in `CANN` development tool guide, which is not described here.

## Tuning Result Viewing

After the tuning starts, a file named `aoe_result_opat_{timestamp}_{pidxxx}.json` will be generated in the working directory to record the tuning process and tuning results. Please refer to [tuning result file analysis](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoe_16_041.html) for specific analysis of this file.

After the tuning is complete, the custom knowledge base will be generated if the conditions are met. If the `TUNE_BANK_PATH` (Environment variable of the knowledge base storage path) is specified, the knowledge base (generated after tuning) will be saved in the specified directory. Otherwise, the knowledge base will be in the following default path `${HOME}/Ascend/latest/data/aoe/custom/graph/${soc_version}`.

## Merging Knowledge Base

After operator tuning, the generated tuning knowledge base supports merging, which is convenient for re-executing, or the other models.(Only the same Ascend AI Processor can be merged). The more specific merging methods can be found in [merging knowledge base](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoepar_16_055.html).

## Notice

Pay attention to the following points when using the AOE tool:

1. The AOE tool can only be used on `Ascend` platform.

2. Ensure that the available disk space in the home directory of the user who performs tuning in the operating environment is at least 20 GB.

3. The AOE tool depends on some third-party software `pciutils`.

4. After the tuning tool is turned on, it is obvious that the compilation time of the perception operator becomes longer.