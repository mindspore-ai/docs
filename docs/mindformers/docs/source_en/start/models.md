# Models

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindformers/docs/source_en/start/models.md)

The current list of models supported by MindFormers is as follows:

<table>
  <thead>
    <tr>
      <th> Models </th>
      <th> Parameters </th>
      <th> Sequence </th>
      <th> Pretraining </th>
      <th> Finetune </th>
      <th> Inference </th>
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
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
     <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
    </tr>
    <tr>
      <td> 70B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 70B </td>
      <td> 8K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/llama3/llama3.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 13B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 14B </td>
      <td> 32K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
    </tr>
    <tr>
      <td> 72B </td>
      <td> 32K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E5%85%A8%E5%8F%82%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen1_5/qwen1_5.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 20B </td>
      <td> 2K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 20B </td>
      <td> 4K </td>
      <td style="text-align: center"> - </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2/internlm2.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
    <tr>
      <td> 34B </td>
      <td> 4K </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E9%A2%84%E8%AE%AD%E7%BB%83"> pretrain </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E5%BE%AE%E8%B0%83"> finetune </a> </td>
      <td> <a href="https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi/yi.md#%E6%8E%A8%E7%90%86"> predict </a> </td>
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
    </tr>
  </tbody>
</table>