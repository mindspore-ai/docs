# Weight Format Conversion

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/weight_conversion.md)

## Overview

MindSpore Transformers provides a unified weight conversion tool that allows model weights to convert between the HuggingFace and MindSpore Transformers formats. This helps you:

- Convert a HuggingFace weight to a MindSpore Transformers one for fine-tuning, evaluation, or inference on MindSpore Transformers.
- Convert the weights trained or fine-tuned using MindSpore Transformers to HuggingFace weights and uses them on other frameworks.

## Conversion Procedure

To perform weight conversion, clone the complete HuggingFace repository of the model to be converted locally, and execute the `mindformers/convert_weight.py` script. This script automatically converts the HuggingFace model weight file into a weight file applicable to MindSpore Transformers. If you want to convert a MindSpore Transformers weight to a HuggingFace one, set `reversed` to `True`.

```shell
python convert_weight.py [-h] --model MODEL [--reversed] --input_path INPUT_PATH  --output_path OUTPUT_PATH [--dtype DTYPE] [--n_head N_HEAD] [--hidden_size HIDDEN_SIZE] [--layers LAYERS] [--is_pretrain IS_PRETRAIN] [--telechat_type TELECHAT_TYPE]
```

### Parameters

- model: model name.
- reversed: converts a MindSpore Transformers weight to the HuggingFace one.
- input_path: path of the HuggingFace weight folder, which points to the downloaded weight file.
- output_path: path for storing the MindSpore Transformers weight file after conversion.
- dtype: weight data type after conversion.
- n_head: takes effect only for the BLOOM model. Set this parameter to `16` when `bloom_560m` is used and to `32` when `bloom_7.1b` is used.
- hidden_size: takes effect only for the BLOOM model. Set this parameter to `1024` when `bloom_560m` is used and to `4096` when `bloom_7.1b` is used.
- layers: number of layers to be converted. This parameter takes effect only for the GPT2 and WizardCoder models.
- is_pretrain: converts the pre-trained weight. This parameter takes effect only for the Swin model.
- telechat_type: version of the TeleChat model. This parameter takes effect only for the TeleChat model.

## Conversion Example

Assume that you have downloaded the [Llama2 model weight](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD) and saved it in the `/home/user/torch_weights` path. To convert it to the MindSpore Transformers weight and save it in the `/home/user/ms_weights` path, run the following command:

```bash
python convert_weight.py --model llama --input_path /home/user/torch_weights --output_path /home/user/ms_weights/llama.ckpt
```

After the preceding steps are performed, the HuggingFace weight is successfully converted to a MindSpore Transformers weight, facilitating model training or inference on MindSpore Transformers.

## Supported Models

| Parameter Value      | Supported models                            |
|-----------|---------------------------------------------|
| llama     | Llama2, Llama3, Llama3.1, CodeLlama         |
| baichuan2 | Baichuan2                                   |
| glm-n     | GLM2, GLM3, GLM3-32K, GLM4                  |
| cogvlm2   | CogVLM2-Video, CogVLM2-Image                |
| qwen      | Qwen, Qwen1.5, Qwen2                        |
| qwenvl    | QwenVL                                      |
| internlm  | InternLM                                    |
| internlm2 | InternLM2                                   |
| yi        | Yi                                          |
| mixtral   | Mixtral                                     |
| deepseek  | DeepSeekCoder, DeepSeekCoder1.5, DeepSeekV2 |
| gpt       | GPT2                                        |
| whisper   | Whisper                                     |

## Developing Weight Conversion for Unsupported Models

1. Add the `convert_weight.py` and `convert_reversed.py` files to the extended model directory.
2. Compile the `convert_pt_to_ms` and `convert_ms_to_pt` weight conversion functions in the files. The function parameters are `input_path`, `output_path`, `dtype`, and an additional parameter `**kwargs`.
3. Add the extended model name and conversion function import paths to the `convert_map` and `reversed_convert_map` dictionaries in the `convert_weight.py` file in the MindSpore Transformers code root directory.
4. Call the `parser.add_argument()` method in the `main` function to add the additional parameter.

## Example of Developing Model Weight Conversion

Llama is used as an example. To convert a HuggingFace weight to a MindSpore Transformers one, define the `convert_pt_to_ms` function in [convert_weight.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_weight.py).

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

To convert a MindSpore Transformers weight to a HuggingFace one, define the `convert_ms_to_pt` function in [convert_reversed.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_reversed.py).

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