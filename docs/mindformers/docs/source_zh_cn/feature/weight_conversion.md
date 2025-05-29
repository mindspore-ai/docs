# 权重格式转换

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/weight_conversion.md)

## 概述

MindSpore Transformers提供了统一的权重转换工具，能够将模型权重在HuggingFace所使用的格式与MindSpore Transformers所使用的格式之间相互转换。这可以帮助用户：

- 将HuggingFace权重转换为MindSpore Transformers权重，在MindSpore Transformers上进行微调、测评或推理。
- 把使用MindSpore Transformers训练或微调得到的权重转换为HuggingFace权重，并在其他框架上使用。

## 转换步骤

要进行权重转换，首先请将待转换模型的HuggingFace仓库完整克隆到本地，然后执行`mindformers/convert_weight.py`脚本。该脚本能够自动将HuggingFace的模型权重文件转换为适用于MindSpore Transformers的权重文件。如若希望将MindSpore Transformers权重转为HuggingFace权重，请将`reversed`设置为`True`。

```shell
python convert_weight.py [-h] --model MODEL [--reversed] --input_path INPUT_PATH  --output_path OUTPUT_PATH [--dtype DTYPE] [--n_head N_HEAD] [--hidden_size HIDDEN_SIZE] [--layers LAYERS] [--is_pretrain IS_PRETRAIN] [--telechat_type TELECHAT_TYPE]
```

### 参数说明

- model：模型名称。
- reversed：将MindSpore Transformers权重转换为HuggingFace权重。
- input_path：HuggingFace权重文件夹的路径，指向已下载的权重文件。
- output_path：转换后MindSpore Transformers权重文件的保存路径。
- dtype：转换后的权重数据类型。
- n_head：只对BLOOM模型生效，使用`bloom_560m`时请设为`16`，使用`bloom_7.1b`时请设为`32`。
- hidden_size：只对BLOOM模型生效，使用`bloom_560m`时请设为`1024`，使用`bloom_7.1b`时请设为`4096`。
- layers：只对GPT2和WizardCoder模型生效，模型被转换的层数。
- is_pretrain：只对Swin模型生效，转换预训练权重。
- telechat_type：只对TeleChat模型生效，TeleChat模型的版本。

## 转换示例

假设用户已经下载了[Llama2模型的权重](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD)，并保存在路径`/home/user/torch_weights`中，用户希望将其转换为MindSpore Transformers权重并保存在路径`/home/user/ms_weights`中，可以使用以下命令：

```bash
python convert_weight.py --model llama --input_path /home/user/torch_weights --output_path /home/user/ms_weights/llama.ckpt
```

通过以上步骤，可将HuggingFace权重成功转换为MindSpore Transformers权重，方便在MindSpore Transformers中继续模型训练或推理。

## 已支持模型

| 参数取值      | 支持模型                                      |
|-----------|-------------------------------------------|
| llama     | Llama2、Llama3、Llama3.1、CodeLlama          |
| baichuan2 | Baichuan2                                 |
| glm-n     | GLM2、GLM3、GLM3-32K、GLM4                   |
| cogvlm2   | CogVLM2-Video、CogVLM2-Image               |
| qwen      | Qwen、Qwen1.5、Qwen2                        |
| qwenvl    | QwenVL                                    |
| internlm  | InternLM                                  |
| internlm2 | InternLM2                                 |
| yi        | Yi                                        |
| mixtral   | Mixtral                                   |
| deepseek  | DeepSeekCoder、DeepSeekCoder1.5、DeepSeekV2 |
| gpt       | GPT2                                      |
| whisper   | Whisper                                   |

## 未支持模型权重转换开发

1. 在扩展模型目录下新增`convert_weight.py`及`convert_reversed.py`文件。
2. 在文件中分别编写`convert_pt_to_ms`及`convert_ms_to_pt`权重转换函数，函数参数为`input_path`、`output_path`、`dtype`及额外参数`**kwargs`。
3. 在MindSpore Transformers代码根目录下`convert_weight.py`文件中的`convert_map`和`reversed_convert_map`字典中加入扩展模型名称及转换函数引入路径。
4. 在`main`函数中通过调用`parser.add_argument()`方法新增额外参数。

## 模型权重转换开发示例

此处以Llama为例。如若希望转换HuggingFace权重至MindSpore Transformers权重，需在[convert_weight.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_weight.py)内定义`convert_pt_to_ms`函数：

```python
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import LlamaForCausalLM
    except:
        raise ImportError(f"Failed to load huggingface checkpoint. Please make sure transformers is available.")

    try:
        model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(input_path))
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]

        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
```

而若是希望转换MindSpore Transformers权重至HuggingFace权重，则需在[convert_reversed.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_reversed.py)内定义`convert_ms_to_pt`函数：

```python
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""
    print(f"Trying to convert mindspore checkpoint in '{input_path}'.", flush=True)
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        name = name_replace(name)
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        if is_lora_param(name):
            name = name.replace('.tk_delta_lora_a', '.lora_A.weight')
            name = name.replace('.tk_delta_lora_b', 'lora_B.weight')
        state_dict[name] = ms2pt(value, dtype)

    torch.save(state_dict, output_path)
    print(f"\rConvert mindspore checkpoint finished, the huggingface checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
```
