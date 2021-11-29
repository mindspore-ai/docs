# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tokenizer"""
import re
import unicodedata
import numpy as np

class Tokenizer():
    """tokenizer"""
    def __init__(self, token_dict, do_lower_case=True):
        self._do_lower_case = do_lower_case
        self.token_to_id = token_dict

        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = '[CLS]'
        self._token_end = '[SEP]'

        self.id_to_token = {value: key for key, value in token_dict.items()}
        self._vocab_size = len(token_dict)


    def tokenize(self, text, maxlen=None):
        """encode"""
        if self._do_lower_case:
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([
                ch for ch in text if unicodedata.category(ch) != 'Mn'
            ])

        src_tokens = []
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                src_tokens.append(ch)
            elif self._is_space(ch) or ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                src_tokens.append(ch)

        tokens = []
        for word in src_tokens:
            tokens.extend(self._word_piece_tokenize(word))

        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            self.truncate_sequence(maxlen, tokens, -index)

        return tokens


    def encode(self, text, maxlen=None):
        """encode"""
        text = self.tokenize(text)
        if maxlen is not None:
            self.truncate_sequence(maxlen, text, pop_index=-2)
        token_ids = self.token_to_ids(text)
        segment_ids = [0] * len(token_ids)
        return token_ids, segment_ids


    def decode(self, ids, tokens=None):
        """decode"""
        tokens = self.id_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text = []
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                text.append(token[2:])
            elif len(token) == 1 and self._is_cjk_character(token):
                text.append(token)
            elif len(token) == 1 and self._is_punctuation(token):
                text.append(token)
                text.append(' ')
            elif i > 0 and self._is_cjk_character(text[-1]):
                text.append(token)
            else:
                text.append(' ')
                text.append(token)

        text = ''.join(text)


        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub(r'(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def token_to_ids(self, tokens):
        ids = []
        unk_ids = self.token_to_id[self._token_unk]
        for token in tokens:
            ids.append(self.token_to_id.get(token, unk_ids))
        return ids


    def id_to_tokens(self, ids):
        tokens = []
        for index in ids:
            tokens.append(self.id_to_token[index])
        return tokens

    def truncate_sequence(self, maxlen, first_sequence, pop_index=-1):
        """截断总长度
        """

        while True:
            total_length = len(first_sequence)
            if total_length <= maxlen:
                break
            if np.random.rand() < 0.5:
                first_sequence.pop(pop_index)
            else:
                first_sequence.pop(1)


    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self.token_to_id:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub_token = word[start:stop]
                if start > 0:
                    sub_token = ''.join(['##', sub_token])
                if sub_token in self.token_to_id:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub_token)
            start = stop

        return tokens

    @staticmethod
    def _is_special(ch):
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _cjk_punctuation():
        a = [u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b',
             u'\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e',
             u'\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e',
             u'\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f',
             u'\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f',
             u'\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002']
        return ''.join(a)
