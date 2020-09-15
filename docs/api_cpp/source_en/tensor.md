# mindspore::tensor

#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/ms_tensor.h)&gt;


## MSTensor

MSTensor defined tensor in MindSpore Lite.

**Constructors & Destructors**
``` 
MSTensor()
```
Constructor of MindSpore Lite MSTensor.

- Returns

    Instance of MindSpore Lite MSTensor.
    
``` 
virtual ~MSTensor()
```

Destructor of MindSpore Lite Model.

**Public Member Functions**

```
virtual TypeId data_type() const
```
Get data type of the MindSpore Lite MSTensor.

> Note: TypeId is defined in [mindspore/mindspore/core/ir/dtype/type_id.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/core/ir/dtype/type_id.h). Only number types in TypeId enum are suitable for MSTensor.

- Returns

    MindSpore Lite TypeId of the MindSpore Lite MSTensor.

```
virtual std::vector<int> shape() const
```

Get shape of the MindSpore Lite MSTensor.

- Returns

    A vector of int as the shape of the MindSpore Lite MSTensor.

```
virtual int DimensionSize(size_t index) const
```

Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.

- Parameters

    - `index`: Define index of dimension returned.

- Returns

    Size of dimension of the MindSpore Lite MSTensor.

```
virtual int ElementsNum() const
```

Get number of element in MSTensor.

- Returns

    Number of element in MSTensor.

```
virtual size_t Size() const
```

Get byte size of data in MSTensor.

- Returns

    Byte size of data in MSTensor.
    

```
virtual void *MutableData() const
```

Get the pointer of data in MSTensor.


> Note: The data pointer can be used to both write and read data in MSTensor.

- Returns

    The pointer points to data in MSTensor.
