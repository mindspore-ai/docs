# 模型库

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindformers/docs/source_zh_cn/start/models.md)

当前MindFormers支持的模型列表如下：

<table>
  <thead>
    <tr>
      <th> 模型 </th>
      <th> 参数 </th>
      <th> 序列 </th>
      <th> 预训练 </th>
      <th> 微调 </th>
      <th> 推理 </th>
      <th> 训练性能（配置/硬件） </th>
      <th> 推理性能（配置/硬件） </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md"> LLama2 </a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 4160 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/pretrain_llama2_7b_bf16.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 332 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/predict_llama2_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 1691 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/finetune_llama2_13b_bf16.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 420 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/predict_llama2_13b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 337 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/finetune_llama2_70b_bf16_32p.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 522 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/predict_llama2_70b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md"> LLama3 </a> </td>
      <td> 8B </td>
      <td> 8K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2581 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/finetune_llama3_8b_8k_800T_A2_64G.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 8K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 337 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/finetune_llama3_70b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/llama3_1.md"> LLama3.1 </a> </td>
      <td> 8B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/llama3_1.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/llama3_1.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2703 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/finetune_llama3_1_8b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 591 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/predict_llama3_1_8b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/llama3_1.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/llama3_1.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 337 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/finetune_llama3_1_70b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 509 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3_1/predict_llama3_1_70b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md"> Baichuan2 </a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 3164 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/finetune_baichuan2_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 521 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/predict_baichuan2_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 1465 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/finetune_baichuan2_13b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 224 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/predict_baichuan2_13b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md"> GLM2 </a> </td>
      <td> 6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 815.2059134 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm2/run_glm2_6b_finetune_800T_A2_64G.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 32.08 tokens/s (seq_length=512) <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm2/predict_glm2_6b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm3.md"> GLM3 </a> </td>
      <td> 6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm3.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm3.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 3450 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm3/finetune_glm3_6b_bf16.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 627 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm3/predict_glm3_6b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm3.md"> GLM3-32K </a> </td>
      <td> 6B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm3.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm3.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 3450 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm3/finetune_glm3_6b_bf16.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 627 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm3/predict_glm3_6b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm4.md"> GLM4 </a> </td>
      <td> 9B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm4.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm4.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2339 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm4/finetune_glm4_9b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 256 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm4/predict_glm4_9b_chat.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_video.md"> CogVLM2-Video </a> </td>
      <td> 13B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_video.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_video.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_image.md"> CogVLM2-Image </a> </td>
      <td> 19B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_image.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md"> Qwen </a> </td>
      <td> 7B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2955 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/finetune_qwen_7b_bf16.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 23 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/predict_qwen_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 1106 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/finetune_qwen_14b_bf16.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 35 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/predict_qwen_14b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md"> Qwen1.5 </a> </td>
      <td> 7B </td>
      <td> 32K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2684 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/finetune_qwen1_5_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 164 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/predict_qwen1_5_7b_chat.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 14B </td>
      <td> 32K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 1452 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/finetune_qwen1_5_14b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 104 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/predict_qwen1_5_14b_chat.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 72B </td>
      <td> 32K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td> 74 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/predict_qwen1_5_72b_chat.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="6"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md"> Qwen2 </a> </td>
      <td> 0.5B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 9555 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/finetune_qwen2_0.5b_32k.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 1907 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/predict_qwen2_0_5b_instruct.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 1.5B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 4363 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/finetune_qwen2_1.5b_32k.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 1160 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/predict_qwen2_1_5b_instruct.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 7B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td> 645 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/predict_qwen2_7b_instruct.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 57B-A14B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 288 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/finetune_qwen2_57b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 57B </td>
      <td> 32K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 72B </td>
      <td> 128K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/qwen2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2026 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/finetune_qwen2_72b_32k.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 252 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/predict_qwen2_72b_instruct.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwenvl/qwenvl.md"> Qwen-VL </a> </td>
      <td> 9.6B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwenvl/qwenvl.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwenvl/qwenvl.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 2587 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/finetune_qwen2_72b_32k.yaml"> 配置 </a> <br> - </td>
      <td> 42 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen2/predict_qwen2_72b_instruct.yaml"> 配置 </a> <br> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md"> InternLM </a> </td>
      <td> 7B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 3250 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/finetune_internlm_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 62 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/predict_internlm_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 20B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td> 296 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/predict_internlm_20b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md"> InternLM2 </a> </td>
      <td> 7B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
    <tr>
      <td> 20B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md"> Yi </a> </td>
      <td> 6B </td>
      <td> 2K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 3324 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/finetune_yi_6b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 31 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/predict_yi_6b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
    <tr>
      <td> 34B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 660 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/finetune_yi_34b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 41 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/predict_yi_34b_chat.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/mixtral/mixtral.md"> Mixtral </a> </td>
      <td> 8x7B </td>
      <td> 32K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/mixtral/mixtral.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/mixtral/mixtral.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/mixtral/mixtral.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/deepseek.md"> DeepSeek Coder </a> </td>
      <td> 33B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/deepseek.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/deepseek.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/deepseek.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 572 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/finetune_deepseek_33b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 292 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/predict_deepseek_33b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek/deepseek.md"> DeepSeek Coder1.5 </a> </td>
      <td> 7B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek1_5/deepseek1_5.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek1_5/deepseek1_5.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 340 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek1_5/finetune_deepseek_coder1_5_7b.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td> 60 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek1_5/predict_deepseek_coder1_5_7b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek2/deepseek2.md"> DeepSeekV2 </a> </td>
      <td> 236B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek2/deepseek2.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek2/deepseek2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 36 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek2/finetune_deepseek2_236B.yaml"> 配置 </a> <br> Atlas 900 A2 PoDc </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/codellama.md"> CodeLlama </a> </td>
      <td> 34B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/codellama.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/codellama.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/codellama.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 667 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/codellama/finetune_codellama_34b_32p.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 139 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/codellama/predict_codellama_34b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md"> GPT2 </a> </td>
      <td> 13B </td>
      <td> 2K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
      <td> 1376 tokens/s/p <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/gpt2/run_gpt2_13b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
      <td> 21 tokens/s <br> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/gpt2/run_gpt2_13b.yaml"> 配置 </a> <br> Atlas 800T A2 </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/whisper.md"> Whisper </a> </td>
      <td> 1.5B </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/whisper.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
      <td style="text-align: center"> - </td>
    </tr>
  </tbody>
</table>