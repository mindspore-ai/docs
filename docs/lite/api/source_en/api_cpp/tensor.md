# mindspore::tensor

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/api/source_en/api_cpp/tensor.md)

## MSTensor

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/ms_tensor.h)&gt;

MSTensor defined tensor in MindSpore Lite.

### Constructors & Destructors

#### MSTensor

```cpp
MSTensor()
```

Constructor of MindSpore Lite MSTensor.

- Returns

    Instance of MindSpore Lite MSTensor.

#### ~MSTensor

```cpp
virtual ~MSTensor()
```

Destructor of MindSpore Lite Model.

### Public Member Functions

#### data_type

```cpp
virtual TypeId data_type() const
```

Get data type of the MindSpore Lite MSTensor.

> TypeId is defined in [mindspore/mindspore/core/ir/dtype/type_id.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/core/ir/dtype/type_id.h). Only number types or kObjectTypeString in TypeId enum are applicable for MSTensor.

- Returns

    MindSpore Lite TypeId of the MindSpore Lite MSTensor.

#### shape

```cpp
virtual std::vector<int> shape() const
```

Get shape of the MindSpore Lite MSTensor.

- Returns

    A vector of int as the shape of the MindSpore Lite MSTensor.

#### ElementsNum

```cpp
virtual int ElementsNum() const
```

Get number of element in MSTensor.

- Returns

    Number of element in MSTensor.

#### Size

```cpp
virtual size_t Size() const
```

Get byte size of data in MSTensor.

- Returns

    Byte size of data in MSTensor.

#### MutableData

```cpp
virtual void *MutableData() const
```

Get the pointer of data in MSTensor.

> The data pointer can be used to both write and read data in MSTensor.
>
> If current MSTensor is from opencl GPU, the data will be mapped from Device to Host using MutableData() interface.

- Returns

    The pointer points to data in MSTensor.
