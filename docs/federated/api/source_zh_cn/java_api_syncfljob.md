# SyncFLJob

<!-- TOC -->

- [SyncFLJob](#syncfljob)
    - [公有成员函数](#公有成员函数)
    - [flJobRun](#fljobrun)
    - [modelInference](#modelinference)
    - [getModel](#getmodel)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_zh_cn/java_api_syncfljob.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

```java
import com.huawei.flclient.SyncFLJob
```

SyncFLJob定义了端侧联邦学习启动接口flJobRun()、端侧推理接口modelInference()、获取云侧最新模型的接口getModel ()。

## 公有成员函数

| **function**                     |
| -------------------------------- |
| public void flJobRun()           |
| public int[] modelInference()    |
| public FLClientStatus getModel() |

## flJobRun

```java
public void flJobRun()
```

启动端侧联邦学习任务，具体使用方法可参考[接口介绍文档](https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_zh_cn/interface_description_federated_client.md)。

## modelInference

```java
public int[] modelInference()
```

启动端侧推理任务，具体使用方法可参考[接口介绍文档](https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_zh_cn/interface_description_federated_client.md)。

- 返回值

  根据输入推理出的标签组成的int[]。

## getModel

```java
public FLClientStatus getModel()
```

获取云侧最新模型，具体使用方法可参考[接口介绍文档](https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_zh_cn/interface_description_federated_client.md)。

- 返回值

  返回getModel请求状态码。
