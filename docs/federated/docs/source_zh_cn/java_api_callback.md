# Callback

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/java_api_callback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.model.Callback
```

Callback定义了端侧联邦学习中用于记录训练、评估和预测不同阶段结果的钩子函数。

## 公有成员函数

| function                    |
| -------------------------------- |
| [abstract Status stepBegin()](#stepbegin) |
| [abstract Status stepEnd()](#stepend)   |
| [abstract Status epochBegin()](#epochbegin) |
| [abstract Status epochEnd()](#epochend) |

## stepBegin

```java
   public abstract Status stepBegin()
```

单步执行前处理函数。

- 返回值

  前处理执行结果状态。

## stepEnd

```java
public abstract Status stepEnd()
```

单步执行后处理函数。

- 返回值

  后处理执行结果状态。

## epochBegin

```java
public abstract Status epochBegin()
```

epoch执行前处理函数。

- 返回值

  前处理执行结果状态。

## epochEnd

```java
public abstract Status epochEnd()
```

epoch执行后处理函数。

- 返回值

  前处理执行结果状态。
