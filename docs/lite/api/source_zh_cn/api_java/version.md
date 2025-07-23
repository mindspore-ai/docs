# Version

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0rc1/docs/lite/api/source_zh_cn/api_java/version.md)

```java
import com.mindspore.config.Version;
```

获取MindSpore Lite 版本信息。

## 公有成员函数

| function                                   | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------------------------------------ |--------|--------|
| [static void init()](#init) | √    | √      |
| [static native String version()](#version) | √    | √      |

## init

```java
public static void init()
```

初始化函数。

## version

```java
public static native String version()
```

获取MindSpore Lite 版本信息。

- 返回值

  MindSpore Lite的版本信息。
