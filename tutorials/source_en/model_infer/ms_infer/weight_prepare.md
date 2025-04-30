# Obtaining and Preparing Large Language Model Weights

[![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/model_infer/ms_infer/weight_prepare.md)

Model weights are the most crucial parameters for large language models and are usually directly related to the model's final performance. Therefore, obtaining effective and reliable model weight files is a very important step in preparing for large language model inference. In general, there are two solutions for obtaining model weight files:

- **Training weights using datasets**: Utilize the training capabilities of the MindSpore framework and a dataset closely related to services to train from scratch or fine-tune a model, then output the model weight file. This approach requires using MindSpore's training capabilities and significant computing resources, making it suitable for scenarios where users have unique datasets. For details, see [mindspore.save_checkpoint](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.save_checkpoint.html#mindspore.save_checkpoint).

- **Obtaining pre-trained model weights from the official websites**: Download pre-trained model configurations, tokenizers, and weight files from the official websites of mainstream models, and use the capabilities of the MindSpore framework to convert these weights into MindSpore's CKPT weight files as the input of large language model inference.

## Obtaining and Converting Weight Files from Hugging Face

This section uses the Llama2-7B large language model as an example to explain how to obtain and convert model weight files into the format required by MindSpore large language models.

### **Downloading the Official Pre-trained Model**

The pre-trained Llama2-7B model can be directly downloaded from Hugging Face's official Hub. Hugging Face provides various download methods, and here we will primarily use the git method for downloading.

```shell
git lfs install
git clone https://huggingface.co/daryl149/llama-2-7b-hf
```

Note: Install the git lfs plug-in beforehand; otherwise, the download may fail.

Once the download is complete, you will see a new directory named llama-2-7b-hf in the current directory. The directory structure is as follows:

```shell
llama-2-7b-hf
│
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00001-of-00002.bin
├── pytorch_model_.bin.index.json
├── README.md
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

In the preceding information, pytorch_model-00001-of-00002.bin and pytorch_model-00001-of-00002.bin are weight files, config.json contains the model configuration, and tokenizer.model is the token mapping table, which are the primary files used in subsequent steps.

### Using the MindSpore Framework to Convert the Weight Files

To convert Hugging Face weight files into MindSpore weight files, perform the following steps:

1. Load the Hugging Face weight files into a list of PyTorch tensors.
2. Convert the PyTorch tensor list into a list of MindSpore tensors.
3. Save the MindSpore tensor list as a MindSpore CKPT weight file.

- **Install the Python dependency package**: Since the conversion involves both Hugging Face and MindSpore, you need to install the respective Python packages, primarily including transformers, torch, and mindspore.

    ```shell
    pip install torch
    pip install mindspore
    pip install transformers
    ```

- **Load the Hugging Face model**: Use the transformers library to load the Llama2 weight files and model, and retrieve the list of weights which is actually a list of PyTorch tensor objects.

    ```python
    import os
    from transformers import LlamaForCausalLM

    hf_ckpt_path="/path/to/huggingface/ckpt"

    model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(hf_ckpt_path))

    hf_weights = model_hf.state_dict()

    for name, value in hf_weights.items():
        print(f"name: {name}")
    ```

Executing this python code will load the weights of Llama2 and print out the names of each weight, indicating that the model has been successfully loaded.

- **Converting torch.Tensor to mindspore.Tensor**: Use NumPy as an intermediary to convert the PyTorch tensor objects into MindSpore tensor objects. In addition to the data, the names of the MindSpore weights differ from those in Hugging Face, so a mapping relationship needs to be recorded.

    - Weight name mapping: Replace the Hugging Face weight names with the MindSpore weight names.

        ```python
        def name_replace(name: str):
            """replace hf param name to ms."""
            name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
            name = name.replace('.self_attn.q_proj.', '.attention.wq.')
            name = name.replace('.self_attn.k_proj.', '.attention.wk.')
            name = name.replace('.self_attn.v_proj.', '.attention.wv.')
            name = name.replace('.self_attn.o_proj.', '.attention.wo.')
            name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
            name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
            name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
            name = name.replace('.input_layernorm.', '.attention_norm.')
            name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
            name = name.replace('.norm.', '.norm_out.')
            return name
        ```

    - Tensor conversion: Convert the PyTorch tensors to a NumPy array, then create the MindSpore tensors using the NumPy array.

        ```python
        import torch
        import mindspore as ms

        def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
            """
            convert torch.Tensor to ms.Tensor with specified dtype
            """
            if value.dtype == torch.bfloat16:
                np_value = value.detach().cpu().to(torch.float32).numpy()
            else:
                np_value = value.detach().numpy()

            if dtype:
                return ms.Tensor(np_value, dtype=dtype)
            return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)
        ```

    - Tensor list conversion: Iterate through all tensors to perform the conversion.

        ```python
        ckpt_list = []
        for name, value in hf_weights.items():
            name = name_replace(name)
            if name == 'norm.weight':
                name = 'norm_out.weight'
            if name[:7] == 'layers.':
                name = name[7:]

            ckpt_list.append({'name': name, 'data': pt2ms(value, ms.float16)})

        print(ckpt_list)
        ```

- **Saving a MindSpore CKPT weight file**: Call the MindSpore API to save tensors as a CKPT weight file.

    ```python
    ms_ckpt_path="/path/to/mindspore/ckpt"
    ms.save_checkpoint(ckpt_list, ms_ckpt_path)
    ```

    Upon successful execution, a CKPT file is generated in `ms_ckpt_path`.

    Combine the preceding code into the same `weight_convert.py` file. The details are as follows:

    ```python
    import os

    import torch
    import mindspore as ms
    from transformers import LlamaForCausalLM

    hf_ckpt_path="/path/to/huggingface/ckpt"
    ms_ckpt_path="/path/to/mindspore/ckpt"

    def name_replace(name: str):
        """replace hf param name to ms."""
        name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
        name = name.replace('.self_attn.q_proj.', '.attention.wq.')
        name = name.replace('.self_attn.k_proj.', '.attention.wk.')
        name = name.replace('.self_attn.v_proj.', '.attention.wv.')
        name = name.replace('.self_attn.o_proj.', '.attention.wo.')
        name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
        name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
        name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
        name = name.replace('.input_layernorm.', '.attention_norm.')
        name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
        name = name.replace('.norm.', '.norm_out.')
        return name

    def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
        """
        convert torch.Tensor to ms.Tensor with specified dtype
        """
        if value.dtype == torch.bfloat16:
            np_value = value.detach().cpu().to(torch.float32).numpy()
        else:
            np_value = value.detach().numpy()

        if dtype:
            return ms.Tensor(np_value, dtype=dtype)
        return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)

    model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(hf_ckpt_path))

    hf_weights = model_hf.state_dict()

    ckpt_list = []
    for name, value in hf_weights.items():
        name = name_replace(name)
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]

        print(f'\rprocessing parameter: {name} {value.shape}    ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, ms.float16)})

    ms.save_checkpoint(ckpt_list, ms_ckpt_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is save in '{ms_ckpt_path}'.", flush=True)
    ```

    After setting the CKPT path, run the script to complete weight conversion.

    ```shell
    python weight_convert.py
    ```
