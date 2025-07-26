# TrainCfg

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_java/train_cfg.md)

```java
import com.mindspore.config.TrainCfg;
```

用于端上模型训练的配置参数。

## 公有成员函数

| function                                   | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------------------------------------ |--------|--------|
| [boolean init()](#init) | ✕     | √      |
| [boolean init(String loss_name)](#init) | ✕     | √      |
| [void free()](#free) | ✕     | √      |
| [boolean addMixPrecisionCfg(boolean dynamicLossScale, float lossScale, int thresholdIterNum)](#addmixprecisioncfg) | ✕     | √      |
| [long getTrainCfgPtr()](#gettraincfgptr) | ✕ | √ |

## init

```java
public boolean init()
```

初始化训练配置。

- 返回值

  初始化状态。

```java
public boolean init(String loss_name)
```

初始化训练配置指定损失函数名称。

- 参数

- `loss_name`:用于分割推理和训练部分的损失函数名称。

- 返回值

  初始化状态。  

## free

```java
public void free()
```

释放训练配置。

## addMixPrecisionCfg

```java
public boolean addMixPrecisionCfg(boolean dynamicLossScale, float lossScale, int thresholdIterNum)
```

将混合精度配置添加到训练配置中。

- 参数
- `dynamicLossScale`: 是动态还是静态损失比例因子。

- `lossScale`:损失比例因子 。

- `thresholdIterNum`: 启用dynamicLossScale时修改损失比例因子的阈值。

- 返回值

  添加状态。

## getTrainCfgPtr

```java
public long getTrainCfgPtr()
```

获取训练配置指针。

- 返回值

  训练配置指针。
