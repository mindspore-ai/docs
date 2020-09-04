# mindspore::session 

## LiteSession

LiteSession defines session in MindSpore Lite for compiling Model and forwarding model.

**Constructors & Destructors**
```
LiteSession()
```
Constructor of MindSpore Lite LiteSession using default value for parameters.
    
```
~LiteSession()
```
Destructor of MindSpore Lite LiteSession.

**Public Member Functions**
```
virtual void BindThread(bool if_bind)
```
Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.

- Parameters

    - `if_bind`: Define whether to bind or unbind threads.

```
virtual int CompileGraph(lite::Model *model)
```
Compile MindSpore Lite model. 

> Note: CompileGraph should be called before RunGraph.

- Parameters

    - `model`: Define the model to be compiled.

- Returns

    STATUS as an error code of compiling graph, STATUS is defined in errorcode.h.

```
virtual std::vector <tensor::MSTensor *> GetInputs() const
```
Get input MindSpore Lite MSTensors of model.

- Returns

    The vector of MindSpore Lite MSTensor.

```   
std::vector <tensor::MSTensor *> GetInputsByName(const std::string &node_name) const
```
Get input MindSpore Lite MSTensors of model by node name.

- Parameters

    - `node_name`: Define node name.

- Returns

    The vector of MindSpore Lite MSTensor.
    
```
virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr)
```
Run session with callback. 
    
> Note: RunGraph should be called after CompileGraph.
    
- Parameters

    - `before`: Define a call_back_function to be called before running each node.

    - `after`: Define a call_back_function called after running each node.

- Returns

    STATUS as an error code of running graph, STATUS is defined in errorcode.h.

```   
virtual std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> GetOutputMapByNode() const
```
Get output MindSpore Lite MSTensors of model mapped by node name.

- Returns

    The map of output node name and MindSpore Lite MSTensor.

```      
virtual std::vector <tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const
```
Get output MindSpore Lite MSTensors of model by node name.

- Parameters

    - `node_name`: Define node name.

- Returns

    The vector of MindSpore Lite MSTensor.

```
virtual std::unordered_map <std::string, mindspore::tensor::MSTensor *> GetOutputMapByTensor() const
```
Get output MindSpore Lite MSTensors of model mapped by tensor name.

- Returns

    The map of output tensor name and MindSpore Lite MSTensor.

```        
virtual std::vector <std::string> GetOutputTensorNames() const
```
Get name of output tensors of model compiled by this session.

- Returns

    The vector of string as output tensor names in order.

```      
virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const
```
Get output MindSpore Lite MSTensors of model by tensor name.

- Parameters

    - `tensor_name`: Define tensor name.

- Returns

    Pointer of MindSpore Lite MSTensor.

```     
virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const
```
Get output MindSpore Lite MSTensors of model by tensor name.
    
- Parameters

    - `tensor_name`: Define tensor name.

- Returns

  Pointer of MindSpore Lite MSTensor.

```      
virtual int Resize(const std::vector <tensor::MSTensor *> &inputs)

```
Resize inputs shape.

- Parameters

    - `inputs`: Define the new inputs shape.

- Returns

    STATUS as an error code of resize inputs, STATUS is defined in errorcode.h.

**Static Public Member Functions**

```
static LiteSession *CreateSession(lite::Context *context)
```
Static method to create a LiteSession pointer.

- Parameters

    - `context`: Define the context of session to be created.

- Returns

    Pointer of MindSpore Lite LiteSession.

        
## CallBackParam

CallBackParam defines input arguments for callBack function.
    
**Attributes**
```
name_callback_param
```
A **string** variable. Node name argument.

```
type_callback_param
```
A **string** variable. Node type argument.