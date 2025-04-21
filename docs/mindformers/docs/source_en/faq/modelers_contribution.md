# Modelers Contribution Guidelines

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/faq/modelers_contribution.md)

## Upload a Model to the Modelers Community

Modelers Community is a model hosting platform where users can upload custom models to [Modelers Community](https://modelers.cn/) for hosting.

### MindSpore Transformers Built-in Models

If the custom model uses a built-in model provided by MindSpore Transformers, i.e. a model whose model code is located under mindformers/models, and no modifications have been made to the model's structure code. You only need to upload the weight file and configuration.

For example, if a user uses MindSpore Transformers built-in ChatGLM2 model, performs fine-tuning training, and wants to share the fine-tuned model weights, uploading the model configuration and weights file is sufficient.

Below is sample code that saves the model configuration and weights:

```python
import mindspore as ms
from mindformers import ChatGLM2Config, ChatGLM2ForConditionalGeneration

config = ChatGLM2Config()
model = ChatGLM2ForConditionalGeneration(config)
ms.load_checkpoint("path/model.ckpt", model)  # Load custom weights

model.save_pretrained("./my_model", save_json=True)
```

The above code runs and saves the config.json file and the mindspore_model.ckpt file (larger weights are automatically split and saved).

After saving, you can use the openmind_hub library for model uploading. See [Model Upload](https://modelers.cn/docs/zh/best-practices/community_contribution/model_contribution.html#%E4%BD%BF%E7%94%A8openmind-hub-client%E4%B8%8A%E4%BC%A0%E6%A8%A1%E5%9E%8B).

```python
import openmind_hub

openmind_hub.upload_folder(
    folder_path="/path/to/local/folder",
    repo_id="username/your-model-name",
    token="your-token",
)
```

An uploaded example can be found in the [OpenLlama model](https://modelers.cn/models/MindSpore-Lab/llama_7b/tree/main) of the Modelers community.

### Custom Models

If the user has customized model code, you need to upload the model code file at the same time and add a mapping in the json configuration file so that it can be imported through the Auto class.

#### Naming Rules

Custom code files uploaded to the community generally have uniform naming rules. Assuming the custom model is named model, its code naming should be as follows:

```text
---- model
    |- configuration_model.py  # Config class code files
    |- modeling_model.py       # Model class code files
    |- tokenization_model.py   # Tokenizer code files
```

#### Adding auto Mapping

In order for the Auto class to be able to find the user-defined model class when it is used, you need to add the auto mapping in the config.json file. The contents of the additions are as follows:

```json
{
  "auto_map": {
    "AutoConfig": "configuration_model.MyConfig",
    "AutoModel": "modeling_model.MyModel",
    "AutoModelForCausalLM": "modeling_model.MyModelForCausalLM",
  },
}
```

If there is a custom tokenizer, the tokenizer needs to be saved:

```python
tokenizer.save_pretrained("./my_model", save_json=True)
```

And add auto mapping to the saved tokenizer_config.json:.

```json
{
  "auto_map": {
    "AutoTokenizer": ["tokenization_model.MyTokenizer", "tokenization_model.MyFastTokenizer"]
  },
}
```

#### Uploading the Model

Model uploading can be done using the openmind_hub library. See [Model Upload](https://modelers.cn/docs/zh/best-practices/community_contribution/model_contribution.html#%E4%BD%BF%E7%94%A8openmind-hub-client%E4%B8%8A%E4%BC%A0%E6%A8%A1%E5%9E%8B).

```python
import openmind_hub

openmind_hub.upload_folder(
    folder_path="/path/to/local/folder",
    repo_id="username/your-model-name",
    token="your-token",
)
```

The uploaded example can be found in the [Model](https://modelers.cn/models/MindSpore-Lab/internlm2-7b/tree/main) of the Modelers community.