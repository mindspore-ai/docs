# Ascend Optimization Engine (AOE)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/debug/aoe.md)&nbsp;&nbsp;

## Overview

Ascend Optimization Engine (AOE) is an automatic tuning tool that makes full use of limited hardware resources to meet the performance requirements of operators and the entire network. The more information about the AOE can be got in [Introduction to AOE](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoe_16_001.html). This document mainly introduces how to use the AOE to tune in MindSpore training scenarios.

## Enabling Tune

1. Online Tune

  Set `aoe_tune_mode` the set_context to enable the AOE tool for online tuning. The value of `aoe_tune_mode` should be in `["online", "offline"]`.

  online: turn on online tuning.

  offline：save GE graph for offline tune. Save GE graph for offline tune. When the path to save the graph is set by `set_context(save_graphs=3, save_graphs_path="path/to/ir/files")`, the graph is saved in the aoe_dump directory of the specified path; otherwise, it is saved in the aoe_dump directory under the current running directory.

  Set `aoe_config` in set_context for tuning configuration. `job_type` is tuning type，and the value should be in `["1", "2"]`，default value is `2`.

  1: subgraph tune.

  2: operator tune.

  Example of online tuning:

  ```python
  import mindspore as ms
  ms.set_context(aoe_tune_mode="online", aoe_config={"job_type": "2"})
  ....
  ```

  After setting the above context, you can start the tuning according to the normal execution of the training script. During the execution of the use case, no operation is required. The result of the model is the result after tuning.

2. Offline Tune

  The Offline Tune is using the dump data (The output description file, and the binary file of operators) of network model (Generate when training network) to tune the operators. The method of Offline Tune and related environment variables can be found in [Offline Tune](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoe_16_023.html) in `CANN` development tool guide, which is not described here.

## Tuning Result Viewing

After the tuning starts, a file named `aoe_result_opat_{timestamp}_{pidxxx}.json` will be generated in the working directory to record the tuning process and tuning results. Please refer to [tuning result file analysis](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoe_16_028.html) for specific analysis of this file.

After the tuning is complete, the custom knowledge base will be generated if the conditions are met. If the `TUNE_BANK_PATH` (Environment variable of the knowledge base storage path) is specified, the knowledge base (generated after tuning) will be saved in the specified directory. Otherwise, the knowledge base will be in the following default path `${HOME}/Ascend/latest/data/aoe/custom/graph/${soc_version}`.

## Merging Knowledge Base

After operator tuning, the generated tuning knowledge base supports merging, which is convenient for re-executing, or the other models.(Only the same Ascend AI Processor can be merged). The more specific merging methods can be found in [merging knowledge base](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/developmenttools/devtool/aoepar_16_055.html).

## Notice

Pay attention to the following points when using the AOE tool:

1. The AOE tool can only be used on `Ascend` platform.

2. Ensure that the available disk space in the home directory of the user who performs tuning in the operating environment is at least 20 GB.

3. The AOE tool depends on some third-party software `pciutils`.

4. After the tuning tool is turned on, it is obvious that the compilation time of the perception operator becomes longer.