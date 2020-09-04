# mindspore::tensor

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

> Note: TypeId is defined in mindspore/mindspore/core/ir/dtype/type_id.h. Only number types in TypeId enum are suitable for MSTensor.

- Returns

    MindSpore Lite TypeId of the MindSpore Lite MSTensor.

```
virtual TypeId set_data_type(TypeId data_type)
```
Set data type for the MindSpore Lite MSTensor.

- Parameters

    - `data_type`: Define MindSpore Lite TypeId to be set in the MindSpore Lite MSTensor.

- Returns

    MindSpore Lite TypeId of the MindSpore Lite MSTensor after set.

```
virtual std::vector<int> shape() const
```

Get shape of the MindSpore Lite MSTensor.

- Returns

    A vector of int as the shape of the MindSpore Lite MSTensor.

```
virtual size_t set_shape(const std::vector<int> &shape)
```
Set shape for the MindSpore Lite MSTensor.

- Parameters

    - `shape`: Define a vector of int as shape to be set into the MindSpore Lite MSTensor.

- Returns

    Size of shape of the MindSpore Lite MSTensor after set.

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
virtual std::size_t hash() const
```

Get hash of the MindSpore Lite MSTensor.

- Returns

    Hash of the MindSpore Lite MSTensor.

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

**Static Public Member Functions**

```
static MSTensor *CreateTensor(TypeId data_type, const std::vector<int> &shape)
```

Static method to create a MSTensor pointer.

> Note: TypeId is defined in mindspore/mindspore/core/ir/dtype/type_id.h. Only number types in TypeId enum are suitable for MSTensor.

- Parameters

    - `data_type`: Define the data type of tensor to be created.

    - `shape`: Define the shape of tensor to be created.

- Returns

    The pointer of MSTensor.