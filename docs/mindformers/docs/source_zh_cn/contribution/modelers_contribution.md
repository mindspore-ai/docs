# 魔乐社区贡献指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/contribution/modelers_contribution.md)

## 上传模型至魔乐社区

魔乐社区是一个模型托管平台，用户可以将自定义模型上传至[魔乐社区](https://modelers.cn/)进行托管。

### MindSpore Transformers内置模型

若用户的自定义模型使用了MindSpore Transformers提供的内置模型，即模型代码位于mindformers/models下的模型，且对模型结构代码未进行任何修改，则只需上传模型的权重文件和配置即可。

如，用户使用MindSpore Transformers的内置ChatGLM2模型，进行了微调训练，想分享微调后的模型权重，那么上传模型配置和权重文件即可。

下面是保存模型配置和权重的示例代码：

```python
import mindspore as ms
from mindformers import ChatGLM2Config, ChatGLM2ForConditionalGeneration

config = ChatGLM2Config()
model = ChatGLM2ForConditionalGeneration(config)
ms.load_checkpoint("path/model.ckpt", model)  # 加载自定义权重

model.save_pretrained("./my_model", save_json=True)
```

上述代码运行后会保存config.json文件和mindspore_model.ckpt文件（较大权重会自动拆分保存）。

保存后可使用openmind_hub库，进行模型上传，可参考[模型上传](https://modelers.cn/docs/zh/best-practices/community_contribution/model_contribution.html#%E4%BD%BF%E7%94%A8openmind-hub-client%E4%B8%8A%E4%BC%A0%E6%A8%A1%E5%9E%8B)。

```python
import openmind_hub

openmind_hub.upload_folder(
    folder_path="/path/to/local/folder",
    repo_id="username/your-model-name",
    token="your-token",
)
```

已上传的例子可参考魔乐社区的[OpenLlama模型](https://modelers.cn/models/MindSpore-Lab/llama_7b/tree/main)。

### 自定义模型

若用户有自定义的模型代码，则需要同时上传模型代码文件，并在json配置文件中添加映射，使其可以通过Auto类导入。

#### 命名规则

上传到社区的自定义代码文件，一般有统一的命名规则。假设自定义模型名为model，其代码命名应当如下：

```text
---- model
    |- configuration_model.py  # Config类代码文件
    |- modeling_model.py       # Model类代码文件
    |- tokenization_model.py   # Tokenizer代码文件
```

#### 添加auto映射

为让Auto类使用时，能够顺利找到用户自定义的模型类，需要在config.json文件中，添加auto映射。添加内容如下：

```json
{
  "auto_map": {
    "AutoConfig": "configuration_model.MyConfig",
    "AutoModel": "modeling_model.MyModel",
    "AutoModelForCausalLM": "modeling_model.MyModelForCausalLM",
  },
}
```

若有自定义tokenizer，则需要保存tokenizer：

```python
tokenizer.save_pretrained("./my_model", save_json=True)
```

并在保存的tokenizer_config.json中添加auto映射:

```json
{
  "auto_map": {
    "AutoTokenizer": ["tokenization_model.MyTokenizer", "tokenization_model.MyFastTokenizer"]
  },
}
```

#### 上传模型

可使用openmind_hub库，进行模型上传，可参考[模型上传](https://modelers.cn/docs/zh/best-practices/community_contribution/model_contribution.html#%E4%BD%BF%E7%94%A8openmind-hub-client%E4%B8%8A%E4%BC%A0%E6%A8%A1%E5%9E%8B)。

```python
import openmind_hub

openmind_hub.upload_folder(
    folder_path="/path/to/local/folder",
    repo_id="username/your-model-name",
    token="your-token",
)
```

已上传的例子可参考魔乐社区的[书生2模型](https://modelers.cn/models/MindSpore-Lab/internlm2-7b/tree/main)。
