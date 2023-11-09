# Running Data Recorder

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/debug/rdr.md)

## Overview

Running Data Recorder (RDR) is a function that MindSpore provides to record data when the training program is running. The data to be recorded will be preset in MindSpore, and if there is a run exception in MindSpore when running the training script, the pre-recorded data in MindSpore will be automatically exported to help locate the cause of the run exception. Different runtime exceptions will export different data, for example, if a `Run task error` exception occurs, it will export information such as computation graph, graph execution order, memory allocation, etc. to help locate the cause of the exception.

> Not all run exceptions will export data, and only some of them are currently supported.
>
> Currently, only graph mode training scenarios are supported to collect CPU/Ascend/GPU related data.

## Usage

### Configuring RDR via Configuration File

1. Create the configuration file `mindspore_config.json`.

    ```json
    {
        "rdr": {
            "enable": true,
            "mode": 1,
            "path": "/path/to/rdr/dir"
        }
    }
    ```

    > enable: Control whether the RDR function is enabled or not.
    >
    > mode: Control RDR data export mode. Set to 1 to export data only when training abnormally terminates, and set to 2 to export data when training abnormally terminates or ends normally.
    >
    > path: Set the path to save data in RDR, only absolute path is supported.

2. Configure the RDR via `context`.

    ```python
    import mindspore as ms
    ms.set_context(env_config_path="./mindspore_config.json")
    ```

### Configuring RDR via Environment Variables

Enable RDR by `export MS_RDR_ENABLE=1`, set the export data mode by `export MS_RDR_MODE=1` or `export MS_RDR_MODE=2`, and then set the root directory path for RDR file export by `export MS_RDR_PATH=/path/to/root/dir`. The RDR file will be saved in the `/path/to/root/dir/rank_{RANK_ID}/rdr/` directory, where `RANK_ID` is the card number in the multi-card training scenario, and the default `RANK_ID=0` in the single-card scenario.

> User-set configuration files take precedence over environment variables.

### Exception Handling

Suppose we train with MindSpore on Ascend 910, the training throws a `Run task error` exception.

At this point we go to the export directory of the RDR file and we can see that there are several files, each representing a type of data. For example, `hwopt_d_before_graph_0.ir` is a computational graph file. Open the file using the text tool to view the calculation diagram and analyze whether it meets expectations.

### Diagnostic Handling

When RDR is turned on and environment variable `export MS_RDR_MODE=2` is set, enter diagnostic mode. At the end of the graph compilation, we can also see the same files saved with exception handling in the export directory of the RDR file.
