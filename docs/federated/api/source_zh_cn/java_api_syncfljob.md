# SyncFLJob

<!-- TOC -->

- [SyncFLJob](#syncfljob)
    - [公有成员函数](#公有成员函数)
    - [flJobRun](#fljobrun)
    - [modelInference](#modelinference)
    - [getModel](#getmodel)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/api/source_zh_cn/java_api_syncfljob.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

```java
import com.huawei.flclient.SyncFLJob
```

SyncFLJob定义了端侧联邦学习启动接口flJobRun()、端侧推理接口modelInference()、获取云侧最新模型的接口getModel ()。

## 公有成员函数

| **function**                                                 |
| ------------------------------------------------------------ |
| public void flJobRun()                                       |
| public int[] modelInference(String flName, String dataPath, String vocabFile, String idsFile, String modelPath) |
| public FLClientStatus getModel(String ip, int port, String flName, String trainModelPath, String inferModelPath, boolean useSSL) |

## flJobRun

```java
public void flJobRun()
```

启动端侧联邦学习任务。

## modelInference

```java
public int[] modelInference(String flName, String dataPath, String vocabFile, String idsFile, String modelPath)
```

启动端侧推理任务。

- 参数

    - `flName`: 联邦学习使用的模型名称, 情感分类任务需设置为”adbert“; 图片分类任务需设置为”lenet“。
    - `dataPath`: 数据集路径，情感分类任务为txt文档格式; 图片分类任务为bin文件格式。
    - `vocabFile`: 数据预处理的词典文件路径， 情感分类任务必须设置；图片分类任务设置为null。
    - `idsFile`: 词典的映射id文件路径， 情感分类任务必须设置；图片分类任务设置为null。
    - `modelPath`: 联邦学习推理模型路径，为.ms文件的绝对路径。

- 返回值

  根据输入推理出的标签组成的int[]。

## getModel

```java
public FLClientStatus getModel(boolean useElb, int serverNum, String ip, int port, String flName, String trainModelPath, String inferModelPath, boolean useSSL)
```

获取云侧最新模型。

- 参数
    - `useElb`: 用于设置是否模拟弹性负载均衡，true代表客户端会将请求随机发给一定范围内的server地址， false客户端的请求会发给固定的server地址，默认为false。
    - `serverNum`: 用于设置模拟弹性负载均衡时可发送请求的server数量，默认为1。
    - `ip`: Server端所启动服务的ip地址，形如“http://10.113.216.106:”。
    - `port`: Server端所启动服务的端口号。
    - `flName`: 联邦学习使用的模型名称, 情感分类任务需设置为”adbert“; 图片分类任务需设置为”lenet“。
    - `trainModelPath`: 联邦学习使用的训练模型路径，为.ms文件的绝对路径。
    - `inferModelPath`: 联邦学习使用的推理模型路径，为.ms文件的绝对路径。
    - `useSSL`: 端云通信是否进行ssl证书认证，默认不进行。
- 返回值

  返回getModel请求状态码。