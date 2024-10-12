# MindIE Service Deployment

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/mindie_deployment.md)

## Introduction

MindIE, full name Mind Inference Engine, is a high-performance inference framework based on Ascend hardware. For more information, please refer to [Official Document](https://www.hiascend.com/software/mindie).

MindFormers are hosted in the model application layer MindIE LLM, and large models in MindFormers can be deployed through MindIE Service.

The model support for MindIE inference can be found in [model repository](https://www.mindspore.cn/mindformers/docs/en/dev/start/models.html).

## Environment Setup

### Software Installation

1. Install MindFormers

   Refer to [MindFormers Official Installation Guide](https://www.mindspore.cn/mindformers/docs/en/dev/quick_start/install.html) for installation.

2. Install MindIE

   Refer to [MindIE Installation Dependencies Documentation (to be released)]() to complete the dependency installation. After that, go to [MindIE Resource Download Center (to be released)]() to download the package and install it.

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

### Modifying MindIE Startup Configuration

Open config.json in the mindie-service directory and modify the server-related configuration.

```bash
cd {MindIE installation directory}/
cd mindie-service/conf
vim config.json
```

where `modelWeightPath` and `backendType` must be modified to configure:

```json
"modelWeightPath": "/path/to/mf_model"
"backendType": "ms"
```

`modelWeightPath` is the model configuration file directory, where model and tokenizer and other related files are placed; `backendType` backend startup method is `ms`.

Other relevant parameters are as follows:

| Optional Configurations          | Value Type | Range of Values             | Configuration Descriptions                                                                                                                       |
| ------------------- | -------- | -------------------- |----------------------------------------------------------------------------------------------------------------------------|
| maxSeqLen           | int32    | Customized by user requirements, >0 | MaxSeqLen. Length of input + length of output <= maxSeqLen, user selects maxSeqLen according to inference scenario                                                                       |
| npuDeviceIds        | list     | Customization by model requirements     | The index value of the NPU card can be used. For example, if the visible card is card 4-7, the value needs to be set to [[0,1,2,3]], corresponding to the index value of this card 4. Setting it to another value reports an out-of-bounds error. The visible card defaults to the machine full-volume card. Resource reference needs to be adjusted according to [CANN Environment Variables](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0029.html).                         |
| worldSize           | int32    | Customization by model requirements     | Total number of cards used                                                                                                                     |
| npuMemSize          | int32    | Customization by Video Memory         | The upper limit of the size (GB) that can be used to request KVCache in the NPU can be calculated according to the actual size of the deployment model: npuMemSize=(total free - weight/mp number)*factor, where the factor is taken as 0.8. Recommended value: 8                                    |
| cpuMemSize          | int32    |  Customization by Memory         | The upper limit of the size (GB) that can be used to request KVCache in CPU is related to the swap function, and the Cache will be released for recalculation when cpuMemSize is insufficient. Recommended value: 5                                                   |
| maxPrefillBatchSize | int32    | [1, maxBatchSize]    | Maximum prefill batch size. maxPrefillBatchSize and maxPrefillTokens will complete the batch if they reach their respective values first. This parameter is mainly used in scenarios where there is a clear need to limit the batch size of the prefill phase, otherwise it can be set to 0 (at this point, the engine will take the maxBatchSize value by default) or the same as maxBatchSize. Required, default value: 50. |
| maxPrefillTokens    | int32    | [5120, 409600]       | At each prefill, the total number of all input tokens in the current batch must not exceed maxPrefillTokens. maxPrefillTokens and maxPrefillBatchSize will complete the current group batch if they reach their respective values first. Required, default value: 8192.                                                                                |
| maxBatchSize        | int32    | [1, 5000]            | Maximum decode batch size, estimated based on model size and NPU graphics memory                                                                                       |
| maxIterTimes        | int32    | [1, maxSeqLen-1]     | The number of decodes that can be performed, i.e. the maximum length of a sentence that can be generated. There is a max_output_length parameter inside the request level, maxIterTimes is a global setting, and max_output_length is taken as the maximum length of the final output.         |

The full set of configuration parameters is available in [MindIE Service Developer's Guide - Quick Start - Configuration Parameter Descriptions (to be released)]()

### Starting Service

```bash
cd /path/to/mindie/latest/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

When `Daemon start success!` appears in the log, it means the service started successfully.

### Viewing Logs

The related logs of MindIE Service:

```bash
tail -f path/to/mindie/mindie-service/latest/mindie-service/output.log
```

The related logs of Python:

```bash
tail -f path/to/mindie/mindie-service/latest/mindie-llm/logs/pythonlog.log
```

## MindIE Service Deployment and Inference Example

The following example installs each component to the default path `/usr/local/Ascend/.` and the model uses `Qwen1.5-72B`.

### Modifying the MindIE Startup Configuration

Open the config.json file in the mindie-service directory and modify the server-related configuration.

```bash
vim /usr/local/Ascend/mindie/1.0.RC3/mindie-service/conf/config.json
```

The configuration of the following fields are as follows:

1. `ModelDeployConfig.ModelConfig.backendType`

   This configuration is the corresponding backend type, required “ms”.

   ```json
   "backendType": "ms"
   ```

2. `ModelDeployConfig.ModelConfig.modelWeightPath`

   This configuration is the model configuration file directory, which holds the model and tokenizer and other related files.

   Using Qwen 1.5-72B as an example, `modelWeightPath` is organized as follows:

   ```reStructuredText
   mf_model
    └── qwen1_5_72b
           ├── config.json                 # Model json configuration file
           ├── vocab.json                  # Model vocab file, corresponding model download on hf
           ├── merges.txt                  # Model merges file, corresponding model download on hf
           ├── predict_qwen1_5_72b.yaml    # Model yaml configuration file
           ├── qwen1_5_tokenizer.py        # Model tokenizer file, copy the corresponding model from the research directory in the mindformers bin
           └── qwen1_5_72b_ckpt_dir        # Model Distributed Weights Folder
   ```

   predict_qwen1_5_72b.yaml needs to be concerned with the following configurations:

   ```yaml
   load_checkpoint: '/mf_model/qwen1_5_72b/qwen1_5_72b_ckpt_dir' # Path to the folder that holds the model distributed weights
   use_parallel: True
   auto_trans_ckpt: False    # Whether to enable automatic weight conversion, with offline slicing set to False
   parallel_config:
     data_parallel: 1
     model_parallel: 4       # Multi-card inference configures the model slicing, which generally corresponds to the number of cards used
     pipeline_parallel: 1
   processor:
     tokenizer:
       vocab_file: "/mf_model/qwen1_5_72b/vocab.json"  # vocab path
       merges_file: "/mf_model/qwen1_5_72b/merges.txt"  # merges path
   ```

   The model's config.json file can be generated using the `save_pretrained` interface. An example of which is shown below:

   ```python
   from mindformers import AutoConfig

   model_config = AutoConfig.from_pretrained("/mf_model/qwen1_5_72b/predict_qwen1_5_72b.yaml")
   model_config.save_pretrained(save_directory="./json/qwen1_5_72b/", save_json=True)
   ```

   Model weights can be downloaded and converted in [Guide to Weight Format Conversion](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html).

   After preparing the model configuration directory, set the parameter `modelWeightPath` to that directory path.

   ```json
   "modelWeightPath": "/mf_model/qwen1_5_72b"
   ```

The final modified config.json is as follows:

```json
{
    "Version": "1.0.0",
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
        "managementIpAddress": "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
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
        "managementTlsCert" : "security/certs/management_server.pem",
        "managementTlsPk" : "security/keys/management_server.key.pem",
        "managementTlsPkPwd" : "security/pass/management_mindie_server_key_pwd.txt",
        "managementTlsCrl" : "security/certs/management_server_crl.pem",
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "pdInterNodeTLSEnabled": false,
        "pdCommunicationPort": 1121,
        "interNodeTlsCaFile": "security/ca/ca.pem",
        "interNodeTlsCert": "security/certs/server.pem",
        "interNodeTlsPk": "security/keys/server.key.pem",
        "interNodeTlsPkPwd": "security/pass/mindie_server_key_pwd.txt",
        "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb"
    },

    "BackendConfig": {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled": false,
        "multiNodesInferPort": 1120,
        "interNodeTLSEnabled": true,
        "interNodeTlsCaFile": "security/ca/ca.pem",
        "interNodeTlsCert": "security/certs/server.pem",
        "interNodeTlsPk": "security/keys/server.key.pem",
        "interNodeTlsPkPwd": "security/pass/mindie_server_key_pwd.txt",
        "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
        "ModelDeployConfig":
        {
            "maxSeqLen" : 16384,
            "maxInputTokenLen" : 16384,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType": "Standard",
                    "modelName" : "Qwen1.5-72B-Chat",
                    "modelWeightPath" : "/mf_model/qwen1_5_72b",
                    "worldSize" : 4,
                    "cpuMemSize" : 16,
                    "npuMemSize" : 16,
                    "backendType": "ms"
                }
            ]
        },

        "ScheduleConfig":
        {
            "templateType": "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 1,
            "maxPrefillTokens" : 16384,
            "prefillTimeMsPerReq" : 60,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 60,
            "decodePolicyType" : 0,

            "maxBatchSize" : 128,
            "maxIterTimes" : 8192,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : true,
            "maxQueueDelayMicroseconds" : 500
        }
    }
}
```

> For testing purposes, the `httpsEnabled` parameter is set to `false`, ignoring subsequent https communication related parameters.

### Starting Service

```bash
cd /usr/local/Ascend/mindie/1.0.RC3/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

The startup was successful, with the following printed message.

```json
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

Examples of MindIE inference for other models can be found in the introduction documentation for each model in [Model Library](https://www.mindspore.cn/mindformers/docs/en/dev/start/models.html).