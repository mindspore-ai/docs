# mindspore::tensor

#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/ms_tensor.h)&gt;


## MSTensor

MSTensor定义了MindSpore Lite中的张量。

**构造函数和析构函数**

```
MSTensor()
```
MindSpore Lite MSTensor的构造函数。

- 返回值

    MindSpore Lite MSTensor的实例。
    
```
virtual ~MSTensor()
```
MindSpore Lite Model的析构函数。

**公有成员函数**

```
virtual TypeId data_type() const
```
获取MindSpore Lite MSTensor的数据类型。

> 注意：TypeId在[mindspore/mindspore/core/ir/dtype/type_id\.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中定义。只有TypeId枚举中的数字类型可用于MSTensor。

- 返回值

    MindSpore Lite MSTensor类的MindSpore Lite TypeId。

```
virtual std::vector<int> shape() const
```
获取MindSpore Lite MSTensor的形状。

- 返回值

    一个包含MindSpore Lite MSTensor形状数值的整型向量。

```
virtual int DimensionSize(size_t index) const
```
通过参数索引获取MindSpore Lite MSTensor的维度的大小。

- 参数

    - `index`: 定义了返回的维度的索引。

- 返回值

    MindSpore Lite MSTensor的维度的大小。

```
virtual int ElementsNum() const
```
获取MSTensor中的元素个数。

- 返回值

    MSTensor中的元素个数

```
virtual size_t Size() const
```
获取MSTensor中的数据的字节数大小。

- 返回值

    MSTensor中的数据的字节数大小。
    

```
virtual void *MutableData() const
```
获取MSTensor中的数据的指针。

> 注意：该数据指针可用于对MSTensor中的数据进行读取和写入。

- 返回值

    指向MSTensor中的数据的指针。
