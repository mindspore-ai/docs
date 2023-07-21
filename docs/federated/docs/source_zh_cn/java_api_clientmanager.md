# ClientManager

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/federated/docs/source_zh_cn/java_api_clientmanager.md)

```java
import com.mindspore.flclient.model.ClientManager
```

ClientManager定义了端侧联邦学习自定义算法模型注册接口。

## 公有成员函数

| function                    |
| -------------------------------- |
| [static void registerClient(Client client)](#registerclient)  |
| [static Client getClient(String name)](#getclient)  |

## registerClient

```java
public static void registerClient(Client client)
```

注册Client对象。

- 参数

    - `client`: 需要注册的Client对象。

## getClient

```java
public static Client getClient(String name)
```

获取Client对象。

- 参数

    - `name`: Client对象名称。
- 返回值

  Client对象。
