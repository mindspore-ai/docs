"""OpenR1-Math-220K Data Handler"""
import numpy as np
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.handler.base_handler import BaseInstructDataHandler


PROMPT_INPUT = r"Please reason step by step, and put your final answer within \boxed{}."
MAX_TOKEN_LENGTH = 20480 # 当前Device内存容量仅支持20K长度


@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class OpenR1Math220kDataHandler(BaseInstructDataHandler):
    """OpenR1-Math-220K Data Handler"""

    def format_func(self, example):
        """format func"""
        # OpenR1-Math-220K的messages列包含了user和assistant的对话内容（含<think>思维链），只需添加system prompt即可
        messages = example.get("messages", "")
        messages = [{'role': 'system', 'content': PROMPT_INPUT}] + messages

        return messages

    def tokenize_func(self, messages):
        """tokenize func"""
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            padding=False,
            truncation=True,
        )
        # labels即把assistant回答的部分tokenize，将user和system部分置成ignore_token_id，合并后得到一个和input_ids等长的序列
        target_index = 0
        for index in range(len(input_ids)):
            if input_ids[index] == 151644 and input_ids[index+1] == 77091:
                target_index = index + 3
                break
        if len(input_ids) > MAX_TOKEN_LENGTH:
            input_ids = input_ids[:MAX_TOKEN_LENGTH] + input_ids[-2:len(input_ids)]
        labels = input_ids[target_index:]
        ignore_length = target_index
        labels = np.concatenate([np.full(ignore_length, self.ignore_token_id), labels])
        assert len(labels) == len(input_ids), f"input_ids length {len(input_ids)} different from labels {len(labels)}"
        return {
            "input_ids": input_ids,
            "labels": labels.tolist(),
        }
