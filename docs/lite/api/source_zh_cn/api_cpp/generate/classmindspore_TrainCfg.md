# Class TrainCfg

\#include &lt;[cfg.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/cfg.h)&gt;

`TrainCfg`MindSpore Lite训练的相关配置参数。

## 构造函数

```cpp
TrainCfg() { this->loss_name_ = "_loss_fn"; }\
TrainCfg(const TrainCfg &rhs) {
    this->loss_name_ = rhs.loss_name_;
    this->mix_precision_cfg_ = rhs.mix_precision_cfg_;
    this->accumulate_gradients_ = rhs.accumulate_gradients_;
  }
```

- 参数

    - `rhs`: 训练配置。

## 析构函数

```cpp
~TrainCfg() = default
```

## 公有成员变量

```cpp
OptimizationLevel optimization_level_ = kO0
```

优化的数据类型。

```cpp
enum OptimizationLevel : uint32_t {
  kO0 = 0,
  kO2 = 2,
  kO3 = 3,
  kAuto = 4,
  kOptimizationType = 0xFFFFFFFF
};
```

```cpp
std::string loss_name_
```

损失节点的名称。

```cpp
MixPrecisionCfg mix_precision_cfg_
```

混合精度配置。

```cpp
bool accumulate_gradients_
```

是否累加梯度。

## 公有成员函数

### GetLossName

```cpp
inline std::vector<std::string> GetLossName() const
```

获得损失名称。

- 返回值

  损失的名称。

### SetLossName

```cpp
inline void SetLossName(const std::vector<std::string> &loss_name)
```

设置损失名称。

- 参数

    - `loss_name`: 损失的名称。
