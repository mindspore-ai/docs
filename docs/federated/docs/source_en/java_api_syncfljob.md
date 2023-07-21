# SyncFLJob

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/federated/docs/source_en/java_api_syncfljob.md)

```java
import com.mindspore.flclient.SyncFLJob
```

SyncFLJob defines the API flJobRun() for starting federated learning on the device, the API modelInference() for inference on the device, the API getModel() for obtaining the latest model on the cloud, and the API stopFLJob() for stopping federated learning training tasks.

## Public Member Functions

| **Function**                     |
| -------------------------------- |
| public FLClientStatus flJobRun() |
| public int[] modelInference()    |
| public FLClientStatus getModel() |
| public void stopFLJob()          |

## flJobRun

```java
public FLClientStatus flJobRun()
```

Starts a federated learning task on the device, for specific usage, please refer to the [interface introduction document](https://www.mindspore.cn/federated/docs/en/master/interface_description_federated_client.html).

- Return value

    The status code of the flJobRun request.

## modelInference

```java
public int[] modelInference()
```

Starts an inference task on the device, for specific usage, please refer to the [interface introduction document](https://www.mindspore.cn/federated/docs/en/master/interface_description_federated_client.html).

- Return value

  int[] composed of the labels inferred from the input.

## getModel

```java
public FLClientStatus getModel()
```

Obtains the latest model on the cloud, for specific usage, please refer to the [interface introduction document](https://www.mindspore.cn/federated/docs/en/master/interface_description_federated_client.html).

- Return value

  The status code of the getModel request.

## stopFLJob

```java
public void stopFLJob()
```

The training task can be stopped by calling this interface during the federated learning training process.

When a thread calls SyncFLJob.flJobRun(), it can use another thread to call SyncFLJob.stopFLJob() to stop the federated learning training task during the federated learning training process.