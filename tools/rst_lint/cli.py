"""overwrite cli.py"""
# Load in our dependencies
from __future__ import absolute_import
import argparse
from collections import OrderedDict
import json
import os
import sys
import re

from docutils.utils import Reporter

from restructuredtext_lint.lint import lint_file

# Generate our levels mapping constant
# DEV: We use an ordered dict for ordering in `--help`
# http://repo.or.cz/docutils.git/blob/422cede485668203abc01c76ca317578ff634b30:/docutils/docutils/utils/__init__.py#l65
WARNING_LEVEL_KEY = 'warning'
LEVEL_MAP = OrderedDict([
    ('debug', Reporter.DEBUG_LEVEL),  # 0
    ('info', Reporter.INFO_LEVEL),  # 1
    (WARNING_LEVEL_KEY, Reporter.WARNING_LEVEL),  # 2
    ('error', Reporter.ERROR_LEVEL),  # 3
    ('severe', Reporter.SEVERE_LEVEL),  # 4
])


# Load in VERSION from standalone file
with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    VERSION = version_file.read().strip()

# Define default contents
DEFAULT_FORMAT = 'text'
DEFAULT_LEVEL_KEY = WARNING_LEVEL_KEY

class Stack:
    """custom a stack."""
    def __init__(self):
        self.__list = []  # 初始化列表，存储栈中元素。因为不需要外界访问，所以私有化。

    def push(self, item):  # 弹出栈顶元素
        self.__list.append(item)

    def peek(self):
        return self.__list[len(self.__list) - 1]

    def pop(self):
        return self.__list.pop()   # 列表的pop方法，默认返回最后一个元素

    def is_empty(self):   # 判断是否栈空
        return self.__list == []

def custom_file(filepath):
    """custom function to get matched_brackets_err"""
    file_name = os.path.basename(filepath)
    try:
        with open(filepath, 'r+', encoding='UTF-8-sig') as f:
            text = f.read()
    except UnicodeDecodeError:
        return []
    matched_brackets_err = []
    result_err = []
    dict_bracket = {')': '(', ']': '[', '）': '（'}
    s = Stack()  # 创建一个栈对象，存储左括号
    # pylint: disable=R1702
    for i in range(len(text)):
        flag = 1
        if text[i] in dict_bracket.values():  # 左括号入栈
            s.push((i, text[i]))  # 因为最后要记录剩余左括号未匹配的情况，所以索引也要存储
        elif text[i] in dict_bracket:  # 遇到右括号
            if s.is_empty() or s.peek()[1] != dict_bracket[text[i]]: # 栈空和缺少对应左括号可以归为一种情况
                if not s.is_empty() and s.peek()[1] != '（' and text[i] != '）': # 去除范围类括号，类似左开右闭
                    if re.findall('[\u4e00-\u9fa5]+', text[s.peek()[0]:i]):
                        flag = 0
                    elif ',' not in text[s.peek()[0]:i]:
                        flag = 0
                    elif ',' in text[s.peek()[0]:i] and text[i-1] == ',':
                        flag = 0
                    elif ',' in text[s.peek()[0]:i] and re.findall(rf',[^\S]+\{text[i]}', text[s.peek()[0]:i+1]):
                        flag = 0
                    elif '）' in text[s.peek()[0]:i] or ')' in text[s.peek()[0]:i]:
                        if text[s.peek()[0]:i].count(')') != text[s.peek()[0]:i].count('(')\
                        or text[s.peek()[0]:i].count('）') != text[s.peek()[0]:i].count('（'):
                            flag = 0
                elif text[i] == '）' or text[i] == ')':
                    try:
                        if isinstance(int(text[i-1]), int):
                            flag = 0
                    except ValueError:
                        matched_brackets_err.append([text[i], i])
                else:
                    matched_brackets_err.append([text[i], i])
            if not s.is_empty() and flag:
                s.pop()  # 右括号匹配时，将左括号出栈
    while not s.is_empty():
        index, item = s.pop()
        matched_brackets_err.append([item, index])

    spec_flag = 1
    if text.count('{') != text.count('}') and file_name.endswith('.rst'):
        spec_doc = re.findall(r'\\left\\\{(?:.|\n|)+?\\right', text)
        for doc in spec_doc:
            if '\\\\' in doc:
                spec_flag = 0
    if matched_brackets_err:
        msg = ''
        for err in matched_brackets_err:
            if not spec_flag and err[0] in ['{', '}']:
                continue
            line = 1
            beg = text[:err[1]].rfind('\n')
            end = text[err[1]:].find('\n')
            err_sentence = text[beg+1:err[1]+end]
            for sentence in text.split('\n'):
                if err_sentence == sentence:
                    break
                line += 1
            msg = f'There are mismatched or missing {err[0]} in the statements'
            result_err.append(["WARNING", filepath, line, msg])
        return result_err
    return []


# Define our CLI function
# pylint: disable=W0622
def _main(paths, format=DEFAULT_FORMAT, stream=sys.stdout, encoding=None, level=LEVEL_MAP[DEFAULT_LEVEL_KEY],
          **kwargs):
    """Define our CLI function"""
    error_dicts = []
    error_occurred = False
    filepaths = []

    for path in paths:
        # Check if the given path is a file or a directory
        if os.path.isfile(path):
            filepaths.append(path)
        elif os.path.isdir(path):
            # Recurse over subdirectories to search for *.rst files
            # pylint: disable=W0612
            for root, subdir, files in os.walk(path):
                for file in files:
                    if file.endswith('.rst'):
                        filepaths.append(os.path.join(root, file))
        else:
            stream.write('Path "{path}" not found as a file nor directory\n'.format(path=path))
            sys.exit(1)
            return

    for filepath in filepaths:
        # Read and lint the file
        unfiltered_file_errors = lint_file(filepath, encoding=encoding, **kwargs)
        file_errors = [err for err in unfiltered_file_errors if err.level >= level]
        custom_errors = custom_file(filepath)

        if file_errors:
            error_occurred = True
            if format == 'text':
                for err in file_errors:
                    # e.g. WARNING readme.rst:12 Title underline too short.
                    stream.write('{err.type} {err.source}:{err.line} {err.message}\n'.format(err=err))
            elif format == 'json':
                error_dicts.extend({
                    'line': error.line,
                    'source': error.source,
                    'level': error.level,
                    'type': error.type,
                    'message': error.message,
                    'full_message': error.full_message,
                } for error in file_errors)

        if custom_errors:
            error_occurred = True
            if format == 'text':
                for err in custom_errors:
                    # e.g. WARNING readme.rst:12 Title underline too short.
                    stream.write('{err[0]} {err[1]}:{err[2]} {err[3]}\n'.format(err=err))


    if format == 'json':
        stream.write(json.dumps(error_dicts))

    if error_occurred:
        sys.exit(2)  # Using 2 for linting failure, 1 for internal error
    else:
        sys.exit(0)  # Success!

def main():
    # Set up options and parse arguments
    parser = argparse.ArgumentParser(description='Lint reStructuredText files. Returns 0 if all files pass linting, '
                                     '1 for an internal error, and 2 if linting failed.')
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('paths', metavar='path', nargs='+', type=str, help='File/folder to lint')
    parser.add_argument('--format', default=DEFAULT_FORMAT, type=str, choices=('text', 'json'),
                        help='Format of the output (default: "{default}")'.format(default=DEFAULT_FORMAT))
    parser.add_argument('--encoding', type=str, help='Encoding of the input file (e.g. "utf-8")')
    parser.add_argument('--level', default=DEFAULT_LEVEL_KEY, type=str, choices=LEVEL_MAP.keys(),
                        help='Minimum error level to report (default: "{default}")'.format(default=DEFAULT_LEVEL_KEY))
    parser.add_argument('--rst-prolog', type=str,
                        help='reStructuredText content to prepend to all files (useful for substitutions)')
    args = parser.parse_args()

    # Convert our level from string to number for `_main`
    args.level = LEVEL_MAP[args.level]

    # Run the main argument
    _main(**args.__dict__)


if __name__ == '__main__':
    main()
