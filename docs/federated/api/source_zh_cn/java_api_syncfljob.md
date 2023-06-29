# SyncFLJob

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/api/source_zh_cn/java_api_syncfljob.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.SyncFLJob
```

SyncFLJob定义了端侧联邦学习启动接口flJobRun()、端侧推理接口modelInference()、获取云侧最新模型的接口getModel()。

## 公有成员函数

| **function**                     |
| -------------------------------- |
| public FLClientStatus flJobRun() |
| public int[] modelInference()    |
| public FLClientStatus getModel() |

## flJobRun

```java
public FLClientStatus flJobRun()
```

启动端侧联邦学习任务，具体使用方法可参考[接口介绍文档](https://www.mindspore.cn/federated/api/zh-CN/r1.3/interface_description_federated_client.html)。

- 返回值

    返回flJobRun请求状态码。

## modelInference

```java
public int[] modelInference()
```

启动端侧推理任务，具体使用方法可参考[接口介绍文档](https://www.mindspore.cn/federated/api/zh-CN/r1.3/interface_description_federated_client.html)。

- 返回值

  根据输入推理出的标签组成的int[]。

## getModel

```java
public FLClientStatus getModel()
```

获取云侧最新模型，具体使用方法可参考[接口介绍文档](https://www.mindspore.cn/federated/api/zh-CN/r1.3/interface_description_federated_client.html)。

- 返回值

  返回getModel请求状态码。