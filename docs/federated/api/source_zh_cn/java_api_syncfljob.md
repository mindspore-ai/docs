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

| **function**                                                 |
| ------------------------------------------------------------ |
| public void flJobRun()                                       |
| public int[] modelInference(String flName, String dataPath, String vocabFile, String idsFile, String modelPath) |
| public FLClientStatus getModel()                             |

## flJobRun

```java
public void flJobRun()
```

启动端侧联邦学习任务，具体使用方法可参考[接口介绍文档](https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_zh_cn/interface_description_federated_client.md)。

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
public FLClientStatus getModel()
```

获取云侧最新模型，具体使用方法可参考[接口介绍文档](https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_zh_cn/interface_description_federated_client.md)。

- 返回值

  返回getModel请求状态码。
