# mindspore::lite

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