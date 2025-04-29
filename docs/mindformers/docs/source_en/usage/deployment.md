# Service Deployment

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/mindie_deployment.md)

## Introduction

MindIE, full name Mind Inference Engine, is a high-performance inference framework based on Ascend hardware. For more information, please refer to [Official Document](https://www.hiascend.com/software/mindie).

MindSpore Transformers are hosted in the model application layer MindIE LLM, and large models in MindSpore Transformers can be deployed through MindIE Service.

The model support for MindIE inference can be found in [model repository](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/models.html).

## Environment Setup

### Software Installation

1. Install MindSpore Transformers

   Refer to [MindSpore Transformers Official Installation Guide](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/quick_start/install.html) for installation.

2. Install MindIE

   Refer to [MindIE Installation Dependencies Documentation](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0010.html) to complete the dependency installation. After that, go to [MindIE Resource Download Center](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann) to download the package and install it.

   MindIE and CANN versions must be matched, version matching relationship is as follows.

   |                                            MindIE                                             |                                                      CANN                                                      |
   |:---------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
   | [2.0.RC1](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann) | [8.1.RC1](https://www.hiascend.com/document/detail/en/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |

### Environment Variables

If the installation path is the default path, you can run the following command to initialize the environment variables of each component.

```bash
# Ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# MindIE
source /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
# MindSpore
export LCAL_IF_PORT=8129
# Networking Configuration
export MS_SCHED_HOST=127.0.0.1     # scheduler node ip address
export MS_SCHED_PORT=8090          # Scheduler node service port
```

> If there are other cards on the machine that have MindIE enabled, you need to be aware of any conflicts with the `MS_SCHED_PORT` parameter. If you get an error on this parameter in the log printout, try again with a different port number.

## Basic Process of Inference Service Deployment

### Preparing Model Files

Create a folder for the specified model related files in the MindIE backend, such as model tokenizer files, yaml configuration files and config files.

```bash
mkdir -p mf_model/qwen1_5_72b
```

Taking Qwen1.5-72B as an example, the folder directory structure is as follows:

```reStructuredText
mf_model
 └── qwen1_5_72b
        ├── config.json                 # Model json configuration file, corresponding model download on Hugging Face
        ├── vocab.json                  # Model vocab file, corresponding model download on Hugging Face
        ├── merges.txt                  # Model merges file, corresponding model download on Hugging Face
        ├── predict_qwen1_5_72b.yaml    # Model yaml configuration file
        ├── qwen1_5_tokenizer.py        # Model tokenizer file, copy the corresponding model from the search directory in the mindformers repository
        └── qwen1_5_72b_ckpt_dir        # Model distributed weight folder
```

predict_qwen1_5_72b.yaml needs to be concerned with the following configuration:

```yaml
load_checkpoint: '/mf_model/qwen1_5_72b/qwen1_5_72b_ckpt_dir' # Path to the folder that holds the model distributed weight
use_parallel: True
auto_trans_ckpt: False    # Whether to enable automatic weight conversion, with offline splitting set to False
parallel_config:
  data_parallel: 1
  model_parallel: 4       # Multi-card inference configures the model splitting, which generally corresponds to the number of cards used
  pipeline_parallel: 1
processor:
  tokenizer:
    vocab_file: "/path/to/mf_model/qwen1_5_72b/vocab.json"  # vocab file absolute path
    merges_file: "/path/to/mf_model/qwen1_5_72b/merges.txt"  # merges file absolute path
```

For model weight downloading and conversions, refer to the [Weight Format Conversion Guide](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/weight_conversion.html).

Required files and configurations may vary from model to model. Refer to the model-specific inference sections in [Model Repository](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/models.html) for details.

### Starting MindIE

#### 1. One-click Start (Recommended)

The mindformers repository provides a one-click pull-up MindIE script with preconfigured environment variable settings and servitization configurations, which allows you to quickly pull up the service by simply entering the directory of the model file.

Go to the `scripts` directory and execute the MindIE startup script:

```shell
cd ./scripts
bash run_mindie.sh --model-name xxx --model-path /path/to/model

# Parameter descriptions
--model-name: Mandatory, set MindIE backend name
--model-path: Mandatory, set model folder path, such as /path/to/mf_model/qwen1_5_72b
--help      : Instructions for using the script
```

View logs:

```bash
tail -f output.log
```

When `Daemon start success!` appears in the log, it means the service started successfully.

#### 2. Customized Startup

The MindIE installation paths are all the default paths `/usr/local/Ascend/.` If you customize the installation path, synchronize the path in the following example.

Open config.json in the mindie-service directory and modify the server-related configuration.

```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

where `modelWeightPath` and `backendType` must be modified to configure:

```bash
"modelWeightPath": "/path/to/mf_model/qwen1_5_72b"
"backendType": "ms"
```

`modelWeightPath` is the model folder created in the previous step, where model and tokenizer and other related files are placed; `backendType` backend startup method is `ms`.

Other relevant parameters are as follows:

| Optional Configurations          | Value Type | Range of Values             | Configuration Descriptions                                                                                                                       |
| ------------------- | -------- | -------------------- |----------------------------------------------------------------------------------------------------------------------------|
| httpsEnabled        | Bool     | True/False           | Whether to enable HTTPS communication security authentication, the default is True. Easy to start, it is recommended to set to False.  |
| maxSeqLen           | int32    | Customized by user requirements, >0 | MaxSeqLen. Length of input + length of output <= maxSeqLen, user selects maxSeqLen according to inference scenario                                                                       |
| npuDeviceIds        | list     | Customization by model requirements     | This configuration item is temporarily disabled. The actual running card is controlled by the visible card environment variable and the worldSize configuration. Resource reference needs to be adjusted by visible card according to [CANN Environment Variables](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0029.html).                         |
| worldSize           | int32    | Customization by model requirements     | The number of cards used for the visible card. Example: ASCEND_RT_VISIBLE_DEVICES=4,0,1,2 and worldSize=2, then take the 4th, 0th card to run.    |
| npuMemSize          | int32    | Customization by Video Memory         | The upper limit of the size (GB) that can be used to request KVCache in the NPU can be calculated according to the actual size of the deployment model: npuMemSize=(total free - weight/mp number)*factor, where the factor is taken as 0.8. Recommended value: 8.                                    |
| cpuMemSize          | int32    |  Customization by Memory         | The upper limit of the size (GB) that can be used to request KVCache in CPU is related to the swap function, and the Cache will be released for recalculation when cpuMemSize is insufficient. Recommended value: 5.                                                   |
| maxPrefillBatchSize | int32    | [1, maxBatchSize]    | Maximum prefill batch size. maxPrefillBatchSize and maxPrefillTokens will complete the batch if they reach their respective values first. This parameter is mainly used in scenarios where there is a clear need to limit the batch size of the prefill phase, otherwise it can be set to 0 (at this point, the engine will take the maxBatchSize value by default) or the same as maxBatchSize. Required, default value: 50. |
| maxPrefillTokens    | int32    | [5120, 409600]       | At each prefill, the total number of all input tokens in the current batch must not exceed maxPrefillTokens. maxPrefillTokens and maxPrefillBatchSize will complete the current group batch if they reach their respective values first. Required, default value: 8192.                                                                                |
| maxBatchSize        | int32    | [1, 5000]            | Maximum decode batch size, estimated based on model size and NPU graphics memory.                                                                                       |
| maxIterTimes        | int32    | [1, maxSeqLen-1]     | The number of decodes that can be performed, i.e. the maximum length of a sentence that can be generated. There is a max_output_length parameter inside the request level, maxIterTimes is a global setting, and max_output_length is taken as the maximum length of the final output.         |

The full set of configuration parameters is available in [MindIE Service Developer's Guide - Quick Start - Configuration Parameter Descriptions](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0285.html).

Run the startup script:

```bash
cd /path/to/mindie/latest/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

When `Daemon start success!` appears in the log, it means the service started successfully.

The related logs of Python:

```bash
export MINDIE_LLM_PYTHON_LOG_TO_FILE=1
export MINDIE_LLM_PYTHON_LOG_PATH=/usr/local/Ascend/mindie/latest/mindie-service/logs/pythonlog.log
tail -f /usr/local/Ascend/mindie/latest/mindie-service/logs/pythonlog.log
```

## MindIE Service Deployment and Inference Example

The following example installs each component to the default path `/usr/local/Ascend/.` and the model uses `Qwen1.5-72B`.

### Preparing Model Files

Take Qwen1.5-72B as an example to prepare the model file directory. For details of the directory structure and configuration, refer to [Preparing Model Files](#preparing-model-files):

```bash
mkdir -p mf_model/qwen1_5_72b
```

### Starting MindIE

#### 1. One-click Start (Recommended)

Go to the `scripts` directory and execute the mindie startup script:

```shell
cd ./scripts
bash run_mindie.sh --model-name qwen1_5_72b --model-path /path/to/mf_model/qwen1_5_72b
```

View log:

```bash
tail -f output.log
```

When `Daemon start success!` appears in the log, it means the service started successfully.

#### 2. Customized Startup

Open config.json in the mindie-service directory and modify the server-related configuration.

```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

The final modified config.json is as follows:

```json
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindservice.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrl" : "security/certs/server_crl.pem",
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrl" : "security/certs/management/server_crl.pem",
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : false,
        "interCommPort" : 1121,
        "interCommTlsCaFile" : "security/grpc/ca/ca.pem",
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrl" : "security/certs/server_crl.pem",
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaFile" : "security/grpc/ca/ca.pem",
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrl" : "security/grpc/certs/server_crl.pem",
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 8192,
            "maxInputTokenLen" : 8192,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "Qwen1.5-72B-Chat",
                    "modelWeightPath" : "/mf_model/qwen1_5_72b",
                    "worldSize" : 4,
                    "cpuMemSize" : 15,
                    "npuMemSize" : 15,
                    "backendType" : "ms"
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 4096,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

> For testing purposes, the `httpsEnabled` parameter is set to `false`, ignoring subsequent https communication related parameters.

Go to the mindie-service directory to start the service:

```bash
cd /usr/local/Ascend/mindie/1.0.RC3/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

The following message is printed, indicating that the startup was successful.

```bash
Daemon start success!
```

### Request Test

After the service has started successfully, you can use the curl command to send a request for verification, as shown in the following example:

```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "I love Beijing, because","stream": false}' http://127.0.0.1:1025/generate
```

The validation is successful with the following returned inference result:

```json
{"generated_text":" it is a city with a long history and rich culture....."}
```

## Model List

Examples of MindIE inference for other models can be found in the introduction documentation for each model in [Model Library](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/start/models.html).