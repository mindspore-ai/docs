# mindspore::lite context

## Allocator

Allocator defines a memory pool for dynamic memory malloc and memory free.

## Context

Context is defined for holding environment variables during runtime.

**Constructors & Destructors**

```
Context()
```

Constructor of MindSpore Lite Context using default value for parameters.

```
Context(int thread_num, std::shared_ptr< Allocator > allocator, DeviceContext device_ctx)
```
Constructor of MindSpore Lite Context using input value for parameters.

- Parameters

    - `thread_num`: Define the work thread number during the runtime.

    - `allocator`: Define the allocator for malloc.

    - `device_ctx`: Define device information during the runtime.

- Returns

    The instance of MindSpore Lite Context.

``` 
 ~Context()
```
Destructor of MindSpore Lite Context.

**Public Attributes**

``` 
float16_priority
``` 
A **bool** value. Defaults to **false**. Prior enable float16 inference.

```
device_ctx_{DT_CPU}
```
A **DeviceContext** struct.

``` 
thread_num_
``` 

An **int** value. Defaults to **2**. Thread number config for thread pool.

``` 
allocator
``` 

A **std::shared_ptr<Allocator>** pointer.

``` 
cpu_bind_mode_ 
``` 

A **CpuBindMode** enum variable. Defaults to **MID_CPU**.     

## ModelImpl
ModelImpl defines the implement class of Model in MindSpore Lite.

## PrimitiveC
Primitive is defined as prototype of operator.

## Model
Model defines model in MindSpore Lite for managing graph.

**Constructors & Destructors**
```
Model()
```

Constructor of MindSpore Lite Model using default value for parameters.

```
virtual ~Model()
```

Destructor of MindSpore Lite Model.

**Public Member Functions**
```
PrimitiveC* GetOp(const std::string &name) const
```
Get MindSpore Lite Primitive by name.

- Parameters 

    - `name`: Define name of primitive to be returned.
    
- Returns 

    The pointer of MindSpore Lite Primitive.

```     
const schema::MetaGraph* GetMetaGraph() const
```
Get graph defined in flatbuffers.

- Returns  

    The pointer of graph defined in flatbuffers.

```
void FreeMetaGraph()
```
Free MetaGraph in MindSpore Lite Model.

**Static Public Member Functions**
```
static Model *Import(const char *model_buf, size_t size)
```
Static method to create a Model pointer.

- Parameters    

    - `model_buf`: Define the buffer read from a model file.   

    - `size`: variable. Define bytes number of model buffer.

- Returns  

    Pointer of MindSpore Lite Model.
        
**Public Attributes**
```
 model_impl_ 
```
The **pointer** of implement of model in MindSpore Lite. Defaults to **nullptr**.

## ModelBuilder
ModelBuilder is defined by MindSpore Lite.

**Constructors & Destructors**
```
ModelBuilder()
```

Constructor of MindSpore Lite ModelBuilder using default value for parameters.

```
virtual ~ModelBuilder()
```

Destructor of MindSpore Lite ModelBuilder.

**Public Member Functions**
```
virtual std::string AddOp(const PrimitiveC &op, const std::vector<OutEdge> &inputs)
```

Add primitive into model builder for model building.

- Parameters    

    - `op`: Define the primitive to be added.   

    - `inputs`: Define input edge of primitive to be added.
    
- Returns   

    ID of the added primitive.

```  
const schema::MetaGraph* GetMetaGraph() const
```
Get graph defined in flatbuffers.

- Returns   

    The pointer of graph defined in flatbuffers.

```
virtual Model *Construct()
```
Finish constructing the model.

## OutEdge
**Attributes**
```
nodeId
```
A **string** variable. ID of a node linked by this edge.

```
outEdgeIndex
```
A **size_t** variable. Index of this edge.

## CpuBindMode
An **enum** type. CpuBindMode defined for holding bind cpu strategy argument.

**Attributes**
``` 
MID_CPU = -1
``` 
Bind middle cpu first.

``` 
HIGHER_CPU = 1
``` 
Bind higher cpu first.

``` 
NO_BIND = 0
``` 
No bind.
    
## DeviceType
An **enum** type. DeviceType defined for holding user's preferred backend.

**Attributes**
``` 
DT_CPU = -1
``` 
CPU device type.

``` 
DT_GPU = 1
``` 
GPU device type.

``` 
DT_NPU = 0
``` 
NPU device type, not supported yet.
    
## DeviceContext

A **struct** . DeviceContext defined for holding DeviceType.

**Attributes**
``` 
type
``` 
A **DeviceType** variable. The device type.

## Version

``` 
std::string Version()
``` 
Global method to get a version string.

- Returns

    The version string of MindSpore Lite.