# Description of Environment Variable Support

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/cloud_docs/source_en/reference/environment_variable_support.md)

This document lists the environment variables supported by MindSpore Lite along with their meanings, and provides the available values and default settings for each environment variable.

|        **Environment Variable**        |                   **Description**                   |     **Allowed Values**      | **Default Value** |
| :------------------------------------: | :-------------------------------------------------: | :-------------------------: | :---------------: |
|                 GLOG_v                 |                  Log level setting                  |           0,1,2,3           |         2         |
|           KEEP_ORIGIN_DTYPE            |        Switch to preserve original data type        |            1, ""            |        ""         |
|            MSLITE_API_TYPE             |          API type selection for benchmark           |           NEW, C            |        NEW        |
|         MINDSPORE_DUMP_CONFIG          | Configuration file for on-device training data dump |   Configuration file path   |        ""         |
|           ASCEND_BACK_POLICY           |           Ascend backend policy selection           |          "ge", ""           |        ""         |
|                RANK_ID                 |             Device card sequence number             |             0-N             |        ""         |
|            ASCEND_DEVICE_ID            |              Ascend hardware device ID              |           0-7, ""           |        ""         |
|             GPU_DEVICE_ID              |               GPU hardware device ID                |           0-7, ""           |        ""         |
|      BENCHMARK_UPDATE_CONFIG_ENV       |        Benchmark tool configuration settings        |           "0", ""           |        ""         |
|          MSLITE_PACKAGE_PATH           |              Test case packaging path               |          File path          |        ""         |
|     MS_ASCEND_CHECK_OVERFLOW_MODE      |              Precision mode selection               | SATURATION_MODE/INFNAN_MODE |    INFNAN_MODE    |
|          DISABLE_REUSE_MEMORY          |        Ascend GE backend memory reuse switch        |          "0", "1"           |         0         |
|      ENABLE_MULTI_BACKEND_RUNTIME      |    Multi-backend heterogeneous computing switch     |         "on", "off"         |        off        |
|         ASCEND_CUSTOM_OPP_PATH         |    Installation path for custom Ascend operators    |          File path          |        ""         |
|            ASCEND_OPP_PATH             |           Path to Ascend operator library           |          File path          |        ""         |
|     MSLITE_ENABLE_CLOUD_INFERENCE      |             Enable cloud-side inference             |          "on", ""           |        ""         |
|               ENABLE_AKG               |               Enable AKG optimization               |          "on", ""           |        ""         |
|         MS_INDEPENDENT_DATASET         |                Use external dataset                 |         "true", ""          |        ""         |
|                OPTIMIZE                |     Enable optimization for MindData scenarios      |         "true", ""          |        ""         |
|             MS_CACHE_HOST              |          Host address for MindData scenarios          |        Host address         |     127.0.0.1     |
|             MS_CACHE_PORT              |          Port number for MindData scenarios           |         Port number         |       50052       |
|               DEVICE_ID                |      Hardware device ID for on-device MindData      |           0-7, ""           |        ""         |
|             MS_CPU_FEATURE             |          CPU instruction set architecture           |           avx512            |        ""         |
| MS_DEV_GRAPH_KERNEL_SPLIT_DEBUG_TUNING |         Graph kernel splitting debug switch         |           on, ""            |        ""         |
|      MS_DEV_DUMP_GRAPH_KERNEL_IR       |                Dump graph kernel IR                 |           on, ""            |        ""         |
|               TIME_STEP                |                Number of iterations                 |           Integer           |        ""         |
|              MAX_ROI_NUM               |    Configure MAX_ROI_NUM according to the implementation of the proposal operator.    |           Integer           |        300        |
|            PARA_GROUP_FILE             |       Communication domain configuration file       |   Configuration file path   |        ""         |
|             MS_ENABLE_HCCL             |          Enable HCCL communication library          |     0 (empty), non-zero     |     0 (empty)     |