# Struct QuantParam

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/types.h)&gt;

一个结构体。QuantParam定义了MSTensor的一组量化参数。

## 公有属性

### bit_num

```cpp
bit_num
```

**int** 类型变量。量化的bit数。

### scale

```cpp
scale
```

**double** 类型变量。

### zero_point

```cpp
zero_point
```

**int32_t** 类型变量。

### min

```cpp
min
```

**double** 类型变量。量化的最小值。

### max

```cpp
max
```

**double** 类型变量。量化的最大值。
