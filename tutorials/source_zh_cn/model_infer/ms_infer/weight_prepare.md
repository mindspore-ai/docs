# 大语言模型权重获取和准备

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/model_infer/ms_infer/weight_prepare.md)

模型权重作为大语言模型最为重要的参数，通常直接和模型最终效果强相关，因此获取有效可靠的模型权重文件，成为准备大语言模型推理非常重要的一步。总的来说，获取模型权重文件有两大类方案：

- **自己通过数据集训练权重**：利用MindSpore框架训练能力，以及业务强相关的数据集，从头训练或者对模型进行微调，然后输出模型的权重文件，该方案需要使用MindSpore训练能力，同时需要较大的计算资源来训练模型，比较适合用户自己数据集比较特殊的场景。[保存模型权重CKPT文件](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.save_checkpoint.html#mindspore.save_checkpoint)。

- **从官网获取预训练模型权重**：从主流模型官方网站上获取预训练好的模型配置、tokenizer和权重文件等，并通过MindSpore框架能力将模型权重转换成MindSpore的CKPT权重文件，作为大语言模型推理的输入。

## 从Hugging Face获取并转换权重文件

本章节以Llama2-7B大语言模型为例，介绍如何获取并转换模型权重文件为MindSpore大语言模型需要的形式。

### **下载官方预训练模型**

Llama2-7B的预训练模型可以直接在Hugging Face的官方Hub上下载获取，Hugging Face官方提供了多种下载方式，此处主要以git方式进行下载。

```shell
git lfs install
git clone https://huggingface.co/daryl149/llama-2-7b-hf
```

注意：执行前需要预先安装好git lfs大文件插件，否则可能会下载失败。

下载完成后，可以看到当前目录下多出了llama-2-7b-hf的目录，其内容目录的结构大致如下：

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

其中：pytorch_model-00001-of-00002.bin和pytorch_model-00001-of-00002.bin为权重文件，config.json是模型配置，tokenizer.model是token映射表，为后续主要使用的文件。

### 利用MindSpore框架转换权重文件

将Hugging Face的权重文件转换成MindSpore的权重文件，可以简单的分解为以下几个步骤：

1. 加载Hugging Face权重文件成为torch的Tensor列表。
2. 将torch的Tensor列表转换为MindSpore的Tensor列表。
3. 将MindSpore的Tensor列表保存为MindSpore的CKPT权重文件。

- **安装python依赖包**：由于需要从Hugging Face转换成MindSpore的权重，因此需要安装两者的python包，主要包含transformers、torch、mindspore。

    ```shell
    pip install torch
    pip install mindspore
    pip install transformers
    ```

- **加载Hugging Face模型**：利用transfomers库加载Llama2的权重文件和模型，并从模型中获取权重列表，实际是一个torch的Tensor对象列表。

    ```python
    import os
    from transformers import LlamaForCausalLM

    hf_ckpt_path="/path/to/huggingface/ckpt"

    model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(hf_ckpt_path))

    hf_weights = model_hf.state_dict()

    for name, value in hf_weights.items():
        print(f"name: {name}")
    ```

执行该python代码，会加载Llama2的权重，并将每个权重的名称打印出来，则表示模型加载成功。

- **torch.Tensor转换为mindspore.Tensor**：利用numpy作为中转，将torch的Tensor对象转换为mindspore的Tensor对象，除了数据外，mindspore权重名称也和Hugging Face的不一样，需要记录一个映射关系。

    - 权重名称映射关系：将Hugging Face的权重名称替换为mindspore的权重名称。

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

    - Tensor转换：将torch的Tensor转为numpy数组，再用numpy数组创建mindspore的Tensor。

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

    - Tensor列表转换：遍历所有Tensor进行转换。

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

- **保存MindSpore的CKPT权重文件**：调用MindSpore的接口将Tensor保存为CKPT权重文件。

    ```python
    ms_ckpt_path="/path/to/mindspore/ckpt"
    ms.save_checkpoint(ckpt_list, ms_ckpt_path)
    ```

    执行成功后，在ms_ckpt_path的路径下会生成一个ckpt文件。

    将上述代码合到同一个weight_convert.py文件里，具体如下：

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

    设置好对应的ckpt路径后，运行脚本就可以完成权重转换。

    ```shell
    python weight_convert.py
    ```
