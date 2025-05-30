.. role:: hidden
    :class: hidden-section

.. currentmodule:: {{ module }}

{% if fullname=="mindformers.AutoConfig" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: get_config_origin_mode, get_support_list, invalid_yaml_name
    :members:

{% elif fullname=="mindformers.modules.OpParallelConfig" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: construct, get_ulysses_cp_num, to_dict, to_diff_dict
    :members:

{% elif fullname=="mindformers.AutoProcessor" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: from_pretrained_origin, get_support_list, invalid_yaml_name, show_support_list
    :members:

{% elif fullname=="mindformers.AutoTokenizer" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: get_class_from_origin_mode, get_support_list, invalid_yaml_name, show_support_list
    :members:

{% elif fullname=="mindformers.core.AdamW" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: clone_state, construct
    :members:

{% elif fullname=="mindformers.core.Came" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: init_came_state, supports_flat_params, supports_memory_efficient_fp16, target, construct
    :members:

{% elif fullname=="mindformers.core.CheckpointMonitor" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: record_last_ckpt_to_json, save_checkpoint, save_checkpoint_network, print_savetime, remove_redundancy
    :members:

{% elif fullname=="mindformers.core.EmF1Metric" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: calc_em_score, calc_f1_score, evaluate_pairs, find_lcs, mixed_segmentation, remove_punctuation
    :members:

{% elif fullname=="mindformers.core.EntityScore" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: compute, get_entities_bios
    :members:

{% elif fullname=="mindformers.core.MFLossMonitor" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: dump_info_to_modelarts, epoch_begin, epoch_end, print_output_info, step_begin, step_end, on_train_step_begin, on_train_step_end, on_train_epoch_begin
    :members:

{% elif fullname=="mindformers.core.ProfileMonitor" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: step_begin, step_end, on_train_step_end, on_train_step_begin
    :members:

{% elif fullname=="mindformers.core.PromptAccMetric" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: calculate_circle
    :members:

{% elif fullname=="mindformers.core.SQuADMetric" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: clear, eval, update
    :members:

{% elif fullname=="mindformers.generation.GenerationConfig" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: from_dict, from_model_config, to_dict, update, from_pretrained
    :members:

{% elif fullname=="mindformers.generation.GenerationMixin" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: add_flags_custom, get_logits_processor, get_logits_warper, prepare_inputs_for_generation, process_logits, slice_incremental_inputs, update_model_kwargs_before_generate, chunk_prefill_infer, prepare_inputs_for_generation_mcore, forward_mcore, infer_mcore, add_flags_custom_mcore
    :members:

{% elif fullname=="mindformers.models.ChatGLM2ForConditionalGeneration" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: add_flags_custom, prepare_inputs_for_generation, prepare_inputs_for_predict_layout, construct, convert_map_dict, convert_name, convert_weight_dict
    :members:

{% elif fullname=="mindformers.models.ChatGLM3Tokenizer" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: build_batch_input, build_chat_input, build_inputs_with_special_tokens, convert_tokens_to_ids, get_vocab, save_vocabulary, tokenize
    :members:

{% elif fullname=="mindformers.models.ChatGLM4Tokenizer" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: build_batch_input, build_chat_input, build_inputs_with_special_tokens, build_single_message, convert_special_tokens_to_ids, convert_tokens_to_string, get_vocab, save_vocabulary
    :members:

{% elif fullname=="mindformers.models.LlamaForCausalLM" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: add_flags_custom, prepare_inputs_for_predict_layout, to_embeddings, construct, prepare_inputs_for_prefill_flatten, convert_map_dict, convert_weight_dict, convert_name, pre_gather_func
    :members:

{% elif fullname=="mindformers.models.LlamaTokenizer" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: convert_tokens_to_string, get_spm_processor, get_vocab, save_vocabulary, tokenize, vocab_size
    :members:

{% elif fullname=="mindformers.models.multi_modal.ModalContentTransformTemplate" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: batch_input_ids, check_modal_builder_tokens, generate_modal_context_positions, stack_data, try_to_batch
    :members:

{% elif fullname=="mindformers.models.PretrainedConfig" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: dict_ms_dtype_to_str, get_config_origin_mode, get_support_list, inverse_parse_config, register_for_auto_class, remove_type, save_config_origin_mode, show_support_list, delete_from_dict
    :members:

{% elif fullname=="mindformers.models.PreTrainedModel" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: base_model, framework, from_pretrained_experimental_mode, from_pretrained_origin_mode, fuse_weight_from_ckpt, get_support_list, is_experimental_mode, load_checkpoint, prepare_inputs_for_predict_layout, remove_type, save_pretrained_experimental_mode, save_pretrained_origin_mode, set_dynamic_inputs, show_support_list, convert_map_dict, convert_weight_dict, convert_name, obtain_qkv_ffn_concat_keys, obtain_name_map, check_pipeline_stage
    :members:

{% elif fullname=="mindformers.models.PreTrainedTokenizer" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: get_special_tokens_mask, tokenize_atom, vocab_size
    :members:

{% elif fullname=="mindformers.models.PreTrainedTokenizerFast" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: backend_tokenizer, can_save_slow_tokenizer, decoder, init_atom_1, init_atom_2, save_vocabulary, vocab_size
    :members:

{% elif fullname=="mindformers.pet.models.LoraModel" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: add_adapter
    :members:

{% elif fullname=="mindformers.pipeline.MultiModalToTextPipeline" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: preprocess
    :members:

{% elif fullname=="mindformers.tools.MindFormerConfig" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: set_value, get_value
    :members:

{% elif fullname=="mindformers.tools.register.MindFormerRegister" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: auto_register, get_instance_type_from_cfg
    :members:

{% elif fullname=="mindformers.Trainer" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: add_callback, get_eval_dataloader, get_last_checkpoint, get_load_checkpoint, get_task_config, get_train_dataloader, init_openmind_repo, pop_callback, push_to_hub, remove_callback, save_model, set_parallel_config, set_recompute_config
    :members:

{% elif fullname=="mindformers.TrainingArguments" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: eval_batch_size, get_device_id, get_device_num, get_rank_id, local_process_index, process_index, set_evaluate, set_push_to_hub, set_testing, to_dict, to_json_string, train_batch_size, world_size
    :members:

{% elif fullname=="mindformers.core.TrainingStateMonitor" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: epoch_begin, epoch_end, step_begin, step_end, on_train_epoch_begin, on_train_step_begin, on_train_step_end, abnormal_global_norm_check
    :members:

{% elif fullname=="mindformers.dataset.CausalLanguageModelDataset" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: perform_token_counting, construct
    :members:

{% elif fullname in ["mindformers.AutoModelForCausalLM", "mindformers.AutoModelForZeroShotImageClassification", "mindformers.AutoModel"] %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: construct
    :members: register, from_config, from_pretrained

{% elif objname[0].istitle() %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: construct
    :members:

{% else %}
{{ fullname | underline }}

.. autofunction:: {{ fullname }}

{% endif %}

..
  autogenerated from _templates/classtemplate.rst
  note it does not have :inherited-members:
