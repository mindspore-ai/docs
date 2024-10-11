# MindIE服务化部署

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/usage/mindie_deployment.md)

## MindIE介绍

MindIE，全称Mind Inference Engine，是基于昇腾硬件的高性能推理框架。详情参考[官方介绍文档](https://www.hiascend.com/software/mindie)。

MindFormers承载在模型应用层MindIE LLM中，通过MindIE Service可以部署MindFormers中的大模型。

MindIE推理的模型支持度可参考[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/models.html)。

## 环境搭建

### 软件安装

1. 安装MindFormers

   参考[MindFormers官方安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/quick_start/install.html)进行安装。

2. 安装MindIE

   参考[MindIE安装依赖文档(待发布)]()完成依赖安装。之后前往[MindIE资源下载中心(待发布)]()下载软件包进行安装。

### 环境变量

若安装路径为默认路径，可以运行以下命令初始化各组件环境变量。

```bash
# Ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# MindIE
source /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
# MindSpore
export LCAL_IF_PORT=8129
# 组网配置
export MS_SCHED_HOST=127.0.0.1     # scheduler节点ip地址
export MS_SCHED_PORT=8090          # scheduler节点服务端口
```

> 若机器上有其他卡已启动MindIE，需要注意`MS_SCHED_PORT`参数是否冲突。日志打印中该参数报错的话，替换为其他端口号重新尝试即可。

## 推理服务部署基本流程

### 修改MindIE启动配置

打开mindie-service目录中的config.json，修改server相关配置。

```bash
cd {MindIE安装目录}/
cd mindie-service/conf
vim config.json
```

其中`modelWeightPath`和`backendType`必须修改配置为：

```json
"modelWeightPath": "/path/to/mf_model"
"backendType": "ms"
```

`modelWeightPath`为模型配置文件目录，放置模型和tokenizer等相关文件；`backendType`后端启动方式为`ms`。

其他相关参数如下：

| 可选配置项          | 取值类型 | 取值范围             | 配置说明                                                                                                                       |
| ------------------- | -------- | -------------------- |----------------------------------------------------------------------------------------------------------------------------|
| maxSeqLen           | int32    | 按用户需求自定义，>0 | 最大序列长度。输入的长度+输出的长度<=maxSeqLen，用户根据自己的推理场景选择maxSeqLen                                                                       |
| npuDeviceIds        | list     | 按模型需求自定义     | 可使用NPU卡的索引值。例如可见卡为4-7卡，该值需设定为[[0,1,2,3]]，对应该4卡的索引值。设置为其他值则会报越界错误。  可见卡默认为机器全量卡，需调整资源参考[CANN环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0029.html)                         |
| worldSize           | int32    | 按模型需求自定义     | 使用的总卡数                                                                                                                     |
| npuMemSize          | int32    | 按显存自定义         | NPU中可以用来申请KVCache的size上限（GB），可按部署模型的实际大小计算得出：npuMemSize=(总空闲-权重/mp数量)*系数，其中系数取0.8。建议值：8                                    |
| cpuMemSize          | int32    | 按内存自定义         | CPU中可以用来申请KVCache的size上限（GB），和swap功能有关，cpuMemSize不足时会将Cache释放进行重计算。建议值：5                                                   |
| maxPrefillBatchSize | int32    | [1, maxBatchSize]    | 最大prefill batch size。maxPrefillBatchSize和maxPrefillTokens谁先达到各自的取值就完成本次组batch。该参数主要是在明确需要限制prefill阶段batch size的场景下使用，否则可以设置为0（此时引擎将默认取maxBatchSize值）或与maxBatchSize值相同。必填，默认值：50。 |
| maxPrefillTokens    | int32    | [5120, 409600]       | 每次prefill时，当前batch中所有input token总数，不能超过maxPrefillTokens。maxPrefillTokens和maxPrefillBatchSize谁先达到各自的取值就完成本次组batch。必填，默认值：8192。                                                                                |
| maxBatchSize        | int32    | [1, 5000]            | 最大decode batch size，根据模型规模和NPU显存估算得出                                                                                       |
| maxIterTimes        | int32    | [1, maxSeqLen-1]     | 可以进行的decode次数，即一句话最大可生成长度。请求级别里面有一个max_output_length参数，maxIterTimes是一个全局设置，与max_output_length取小作为最终output的最长length         |

全量配置参数可查阅 [MindIE Service开发指南-快速开始-配置参数说明(待发布)]()

### 启动服务

```bash
cd /path/to/mindie/latest/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

当log日志中出现`Daemon start success!`，表示服务启动成功。

### 查看日志

MindIE Service相关日志：

```bash
tail -f path/to/mindie/mindie-service/latest/mindie-service/output.log
```

Python相关日志：

```bash
tail -f path/to/mindie/mindie-service/latest/mindie-llm/logs/pythonlog.log
```

## MindIE服务化部署及推理示例

以下例子各组件安装路径均为默认路径`/usr/local/Ascend/.` ， 模型使用`Qwen1.5-72B`。

### 修改MindIE启动配置

打开mindie-service目录中的config.json文件，修改server相关配置。

```bash
vim /usr/local/Ascend/mindie/1.0.RC3/mindie-service/conf/config.json
```

需要关注以下字段的配置

1. `ModelDeployConfig.ModelConfig.backendType`

   该配置为对应的后端类型，必填"ms"。

   ```json
   "backendType": "ms"
   ```

2. `ModelDeployConfig.ModelConfig.modelWeightPath`

   该配置为模型配置文件目录，放置模型和tokenizer等相关文件。

   以Qwen1.5-72B为例，`modelWeightPath`的组织结构如下：

   ```reStructuredText
   mf_model
    └── qwen1_5_72b
           ├── config.json                 # 模型json配置文件
           ├── vocab.json                  # 模型vocab文件，hf上对应模型下载
           ├── merges.txt                  # 模型merges文件，hf上对应模型下载
           ├── predict_qwen1_5_72b.yaml    # 模型yaml配置文件
           ├── qwen1_5_tokenizer.py        # 模型tokenizer文件，从mindformers仓中research目录下找到对应模型复制
           └── qwen1_5_72b_ckpt_dir        # 模型分布式权重文件夹
   ```

   predict_qwen1_5_72b.yaml需要关注以下配置：

   ```yaml
   load_checkpoint: '/mf_model/qwen1_5_72b/qwen1_5_72b_ckpt_dir' # 为存放模型分布式权重文件夹路径
   use_parallel: True
   auto_trans_ckpt: False    # 是否开启自动权重转换，离线切分设置为False
   parallel_config:
     data_parallel: 1
     model_parallel: 4       # 多卡推理配置模型切分，一般与使用卡数一致
     pipeline_parallel: 1
   processor:
     tokenizer:
       vocab_file: "/mf_model/qwen1_5_72b/vocab.json"  # vocab文件路径
       merges_file: "/mf_model/qwen1_5_72b/merges.txt"  # merges文件路径
   ```

   模型的config.json文件可以使用`save_pretrained`接口生成，示例如下：

   ```python
   from mindformers import AutoConfig

   model_config = AutoConfig.from_pretrained("/mf_model/qwen1_5_72b/predict_qwen1_5_72b.yaml")
   model_config.save_pretrained(save_directory="./json/qwen1_5_72b/", save_json=True)
   ```

   模型权重下载和转换可参考 [权重格式转换指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)。

   准备好模型配置目录后，设置参数`modelWeightPath`为该目录路径。

   ```json
   "modelWeightPath": "/mf_model/qwen1_5_72b"
   ```

最终修改完后的config.json如下：

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

> 为便于测试，`httpsEnabled`参数设置为`false`，忽略后续https通信相关参数。

### 启动服务

```bash
cd /usr/local/Ascend/mindie/1.0.RC3/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

打印如下信息，启动成功。

```json
Daemon start success!
```

### 请求测试

服务启动成功后，可使用curl命令发送请求验证，样例如下：

```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "I love Beijing, because","stream": false}' http://127.0.0.1:1025/generate
```

返回推理结果验证成功：

```json
{"generated_text":" it is a city with a long history and rich culture....."}
```

## 模型列表

其他模型的MindIE推理示例可参考[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/models.html)中的各模型的介绍文档。