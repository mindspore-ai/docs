# mindspore::lite

#include &lt;[context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/context.h)&gt;

#include &lt;[model.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/model.h)&gt;

#include &lt;[version.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/version.h)&gt;


## Allocator

Allocator类定义了一个内存池，用于动态地分配和释放内存。

## Context

Context类用于保存执行中的环境变量。

**构造函数和析构函数**

```
Context()
```

用默认参数构造MindSpore Lite Context 对象。

```
Context(int thread_num, std::shared_ptr<Allocator> allocator, DeviceContext device_ctx)
```

根据输入参数构造MindSpore Lite Context 对象。

- 参数

    - `thread_num`: 定义了执行线程数。

    - `allocator`: 定义了内存分配器。

    - `device_ctx`: 定义了设备信息。

- 返回值
  
    MindSpore Lite Context 指针。

```
~Context()
```

MindSpore Lite Context 的析构函数。

**公有属性**

```
float16_priority
```

**bool** 值，默认为**false**，用于使能float16 推理。

```
device_ctx_{DT_CPU}
```

[**DeviceContext**](https://www.mindspore.cn/lite/docs/zh-CN/master/apicc/lite.html#devicecontext)结构体。用于设置设备信息。

```
thread_num_
```

**int** 值，默认为**2**，设置线程数。

```
allocator
```

指针类型，指向内存分配器[**Allocator**](https://www.mindspore.cn/lite/docs/zh-CN/master/apicc/lite.html#allocator)的指针。

```
cpu_bind_mode_ 
```

[**CpuBindMode**](https://www.mindspore.cn/lite/docs/zh-CN/master/apicc/lite.html#cpubindmode)枚举类型，默认为**MID_CPU**。 

## ModelImpl

ModelImpl定义了MindSpore Lite中的Model的实现类。

## PrimitiveC

PrimitiveC定义为算子的原型。

## Model

Model定义了MindSpore Lite中的模型，便于计算图管理。

**构造函数和析构函数**

```
Model()
```

MindSpore Lite Model的构造函数，使用默认参数。

```
virtual ~Model()
```

MindSpore Lite Model的析构函数。

**公有成员函数**

```
PrimitiveC* GetOp(const std::string &name) const
```

通过名称获取MindSpore Lite的Primitive对象。

- 参数 

  - `name`: 定义了所要返回的Primitive对象名。

- 返回值 

  指向MindSpore Lite Primitive的指针。

```     
const schema::MetaGraph* GetMetaGraph() const
```

获取在flatbuffers中定义的图。

- 返回值  

  指向flatbuffers中定义的图的指针。

```
void FreeMetaGraph()
```

释放MindSpore Lite Model中的MetaGraph。

**静态公有成员函数**

```
static Model *Import(const char *model_buf, size_t size)
```

创建Model指针的静态方法。

- 参数    

  - `model_buf`: 定义了读取模型文件的缓存区。   

  - `size`: 定义了模型缓存区的字节数。

- 返回值  

  指向MindSpore Lite的Model的指针。
      

**公有属性**

```
 model_impl_ 
```

指向MindSpore Lite模型实现的指针型变量。默认为**nullptr**。

## ModelBuilder

ModelBuilder定义了MindSpore Lite中的模型构建器。

**构造函数和析构函数**

```
ModelBuilder()
```

MindSpore Lite ModelBuilder的构造函数，使用默认参数。

```
virtual ~ModelBuilder()
```

MindSpore Lite ModelBuilder的析构函数。

**公有成员函数**

```
virtual std::string AddOp(const PrimitiveC &op, const std::vector<OutEdge> &inputs)
```

向模型构建器中添加Primitive对象，用于构建模型。

- 参数    

  - `op`: 定义了添加的Primitive对象。   

  - `inputs`: 一个[**OutEdge**](https://www.mindspore.cn/lite/docs/zh-CN/master/apicc/lite.html#outedge)结构体的向量, 定义了添加的Primitive对象的输入的边。

- 返回值   

  添加的Primitive对象的ID。

```  
const schema::MetaGraph* GetMetaGraph() const
```

获取在flatbuffers中所定义的图。

- 返回值   

  指向flatbuffers中所定义的图的指针。

```
virtual Model *Construct()
```

结束模型构建。

## OutEdge

一个结构体。OutEdge定义了计算图的边。

**属性**

```
nodeId
```

**string**类型变量。 被当前边所连接的节点的ID。

```
outEdgeIndex
```

**size_t**类型变量。 当前边的索引。

## CpuBindMode
枚举类型，设置cpu绑定策略。

**属性**

```
MID_CPU = -1
```

优先中等CPU绑定策略。

```
HIGHER_CPU = 1
```

优先高级CPU绑定策略。

```
NO_BIND = 0
```

不绑定。

## DeviceType
枚举类型，设置设备类型。

**属性**

```
DT_CPU = -1
```

设备为CPU。

```
DT_GPU = 1
```

设备为GPU。

```
DT_NPU = 0
```

设备为NPU，暂不支持。

## DeviceContext

定义设备类型的结构体。

**属性**

```
type
```

[**DeviceType**](https://www.mindspore.cn/lite/docs/zh-CN/master/apicc/lite.html#devicetype) 变量。设备类型。

## Version

```
std::string Version()
```
全局方法，用于获取版本的字符串。

- 返回值

    MindSpore Lite版本的字符串。