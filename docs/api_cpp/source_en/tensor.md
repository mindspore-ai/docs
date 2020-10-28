# mindspore::tensor

<a href="https://gitee.com/mindspore/docs/blob/master/docs/api_cpp/source_en/tensor.md" target="_blank"><img src="./_static/logo_source.png"></a>

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/ms_tensor.h)&gt;

## MSTensor

MSTensor defined tensor in MindSpore Lite.

### Constructors & Destructors

```cpp
MSTensor()
```

Constructor of MindSpore Lite MSTensor.

- Returns

    Instance of MindSpore Lite MSTensor.

```cpp
virtual ~MSTensor()
```

Destructor of MindSpore Lite Model.

### Public Member Functions

```cpp
virtual TypeId data_type() const
```

Get data type of the MindSpore Lite MSTensor.

> TypeId is defined in [mindspore/mindspore/core/ir/dtype/type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h). Only number types or kObjectTypeString in TypeId enum are suitable for MSTensor.

- Returns

    MindSpore Lite TypeId of the MindSpore Lite MSTensor.

```cpp
virtual std::vector<int> shape() const
```

Get shape of the MindSpore Lite MSTensor.

- Returns

    A vector of int as the shape of the MindSpore Lite MSTensor.

```cpp
virtual int DimensionSize(size_t index) const
```

Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.

- Parameters

    - `index`: Define index of dimension returned.

- Returns

    Size of dimension of the MindSpore Lite MSTensor.

```cpp
virtual int ElementsNum() const
```

Get number of element in MSTensor.

- Returns

    Number of element in MSTensor.

```cpp
virtual size_t Size() const
```

Get byte size of data in MSTensor.

- Returns

    Byte size of data in MSTensor.

```cpp
virtual void *MutableData() const
```

Get the pointer of data in MSTensor.

> The data pointer can be used to both write and read data in MSTensor.

- Returns

    The pointer points to data in MSTensor.
