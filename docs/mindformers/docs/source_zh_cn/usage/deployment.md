# 服务化部署

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/usage/mindie_deployment.md)

## MindIE介绍

MindIE，全称Mind Inference Engine，是基于昇腾硬件的高性能推理框架。详情参考[官方介绍文档](https://www.hiascend.com/software/mindie)。

MindSpore Transformers承载在模型应用层MindIE LLM中，通过MindIE Service可以部署MindSpore Transformers中的大模型。

MindIE推理的模型支持度可参考[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)。

## 环境搭建

### 软件安装

1. 安装MindSpore Transformers

   参考[MindSpore Transformers官方安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/quick_start/install.html)进行安装。

2. 安装MindIE

   参考[MindIE安装依赖文档](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0010.html)完成依赖安装。之后前往[MindIE资源下载中心](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)下载软件包进行安装。

   MindIE与CANN版本必须配套使用，其版本配套关系如下所示。

   |                                            MindIE                                             |                                                      CANN                                                      |
   |:---------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
   | [2.0.RC1](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann) | [8.1.RC1](https://www.hiascend.com/document/detail/en/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |

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

### 准备模型文件

创建一个文件夹，用于存放MindIE后端的指定模型相关文件，如模型tokenizer文件、yaml配置文件和config文件等。

```bash
mkdir -p mf_model/qwen1_5_72b
```

以Qwen1.5-72B为例，文件夹目录结构如下：

```reStructuredText
mf_model
 └── qwen1_5_72b
        ├── config.json                 # 模型json配置文件，Hugging Face上对应模型下载
        ├── vocab.json                  # 模型vocab文件，Hugging Face上对应模型下载
        ├── merges.txt                  # 模型merges文件，Hugging Face上对应模型下载
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
    vocab_file: "/path/to/mf_model/qwen1_5_72b/vocab.json"  # vocab文件绝对路径
    merges_file: "/path/to/mf_model/qwen1_5_72b/merges.txt"  # merges文件绝对路径
```

模型权重下载和转换可参考 [权重格式转换指南](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/weight_conversion.html)。

不同模型的所需文件和配置可能会有差异，详情参考[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)中具体模型的推理章节。

### 启动MindIE

#### 1. 一键启动（推荐）

mindformers仓上提供一键拉起MindIE脚本，脚本中已预置环境变量设置和服务化配置，仅需输入模型文件目录后即可快速拉起服务。

进入`scripts`目录下，执行MindIE启动脚本：

```shell
cd ./scripts
bash run_mindie.sh --model-name xxx --model-path /path/to/model

# 参数说明
--model-name: 必传，设置MindIE后端名称
--model-path：必传，设置模型文件夹路径，如/path/to/mf_model/qwen1_5_72b
--help      : 脚本使用说明
```

查看日志：

```bash
tail -f output.log
```

当log日志中出现`Daemon start success!`，表示服务启动成功。

#### 2. 自定义启动

MindIE安装路径均为默认路径`/usr/local/Ascend/.` 如自定义安装路径，同步修改以下例子中的路径。

打开mindie-service目录中的config.json，修改server相关配置。

```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

其中`modelWeightPath`和`backendType`必须修改配置为：

```bash
"modelWeightPath": "/path/to/mf_model/qwen1_5_72b"
"backendType": "ms"
```

`modelWeightPath`为上一步创建出的模型文件夹，放置模型和tokenizer等相关文件；`backendType`后端启动方式必须为`ms`。

其他相关参数如下：

| 可选配置项          | 取值类型 | 取值范围             | 配置说明                                                                                                                                                                             |
| ------------------- | -------- | -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| httpsEnabled        | Bool     | True/False           | 是否开启HTTPS通信安全认证，默认为True。便于启动，建议设置为False。                                                                                                                                         |
| maxSeqLen           | int32    | 按用户需求自定义，>0 | 最大序列长度。输入的长度+输出的长度<=maxSeqLen，用户根据自己的推理场景选择maxSeqLen。                                                                                                                            |
| npuDeviceIds        | list     | 按模型需求自定义     | 此配置项暂不生效。实际运行的卡由可见卡环境变量和worldSize配置控制。可见卡需调整资源参考[CANN环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0029.html)。    |
| worldSize           | int32    | 按模型需求自定义     | 可见卡的使用卡数。例：ASCEND_RT_VISIBLE_DEVICES=4,0,1,2且worldSize=2，则取第4，0卡运行。                                                                                                              |
| npuMemSize          | int32    | 按显存自定义         | NPU中可以用来申请KVCache的size上限（GB），可按部署模型的实际大小计算得出：npuMemSize=(总空闲-权重/mp数量)*系数，其中系数取0.8。建议值：8。                                                                                         |
| cpuMemSize          | int32    | 按内存自定义         | CPU中可以用来申请KVCache的size上限（GB），和swap功能有关，cpuMemSize不足时会将Cache释放进行重计算。建议值：5。                                                                                                        |
| maxPrefillBatchSize | int32    | [1, maxBatchSize]    | 最大prefill batch size。maxPrefillBatchSize和maxPrefillTokens谁先达到各自的取值就完成本次组batch。该参数主要是在明确需要限制prefill阶段batch size的场景下使用，否则可以设置为0（此时引擎将默认取maxBatchSize值）或与maxBatchSize值相同。必填，默认值：50。 |
| maxPrefillTokens    | int32    | [5120, 409600]       | 每次prefill时，当前batch中所有input token总数，不能超过maxPrefillTokens。maxPrefillTokens和maxPrefillBatchSize谁先达到各自的取值就完成本次组batch。必填，默认值：8192。                                                    |
| maxBatchSize        | int32    | [1, 5000]            | 最大decode batch size，根据模型规模和NPU显存估算得出。                                                                                                                                            |
| maxIterTimes        | int32    | [1, maxSeqLen-1]     | 可以进行的decode次数，即一句话最大可生成长度。请求级别里面有一个max_output_length参数，maxIterTimes是一个全局设置，与max_output_length取小作为最终output的最长length。                                                              |

全量配置参数可查阅 [MindIE Service开发指南-快速开始-配置参数说明](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0285.html)。

运行启动脚本：

```bash
cd /path/to/mindie/latest/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

当log日志中出现`Daemon start success!`，表示服务启动成功。

Python相关日志：

```bash
export MINDIE_LLM_PYTHON_LOG_TO_FILE=1
export MINDIE_LLM_PYTHON_LOG_PATH=/usr/local/Ascend/mindie/latest/mindie-service/logs/pythonlog.log
tail -f /usr/local/Ascend/mindie/latest/mindie-service/logs/pythonlog.log
```

## MindIE服务化部署及推理示例

以下例子各组件安装路径均为默认路径`/usr/local/Ascend/.` ， 模型使用`Qwen1.5-72B`。

### 准备模型文件

以Qwen1.5-72B为例，准备模型文件目录。目录结构及配置详情可参考[准备模型文件](#准备模型文件)：

```bash
mkdir -p mf_model/qwen1_5_72b
```

### 启动MindIE

#### 1. 一键启动（推荐）

进入`scripts`目录下，执行mindie启动脚本：

```shell
cd ./scripts
bash run_mindie.sh --model-name qwen1_5_72b --model-path /path/to/mf_model/qwen1_5_72b
```

查看日志：

```bash
tail -f output.log
```

当log日志中出现`Daemon start success!`，表示服务启动成功。

#### 2. 自定义启动

打开mindie-service目录中的config.json，修改server相关配置。

```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

修改完后的config.json如下：

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

> 为便于测试，`httpsEnabled`参数设置为`false`，忽略后续https通信相关参数。

进入mindie-service目录启动服务：

```bash
cd /usr/local/Ascend/mindie/1.0.RC3/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

打印如下信息，启动成功。

```bash
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

其他模型的MindIE推理示例可参考[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)中的各模型的介绍文档。