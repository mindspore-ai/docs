# mindspore::lite
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