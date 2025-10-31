# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import glob
import os
import re
import shutil
import sys
from sphinx.ext import autodoc as sphinx_autodoc
import sphinx.ext.autosummary.generate as g
from pathlib import Path

sys.path.append(os.path.abspath('../_ext'))

# -- Project information -----------------------------------------------------

project = 'MindSpore Golden Stick'
copyright = 'MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
myst_enable_extensions = ["dollarmath", "amsmath"]


myst_heading_anchors = 5
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
mathjax_path = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/mathjax/MathJax-3.2.2/es5/tex-mml-chtml.js'

mathjax_options = {
    'async':'async'
}

nbsphinx_requirejs_path = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'

nbsphinx_requirejs_options = {
    "crossorigin": "anonymous",
    "integrity": "sha256-1fEPhSsRKlFKGfK3eO710tEweHh1fwokU5wFGDHO+vg="
}

smartquotes_action = 'De'

exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

autosummary_generate_overwrite = False

html_search_language = 'zh'

html_search_options = {'dict': '../../../resource/jieba.txt'}

# -- Options for HTML output -------------------------------------------------

# Reconstruction of sphinx auto generated document translation.

language = 'zh_CN'
locale_dirs = ['../../../../resource/locale/']
gettext_compact = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

import sphinx_rtd_theme
layout_target = os.path.join(os.path.dirname(sphinx_rtd_theme.__file__), 'layout.html')
layout_src = '../../../../resource/_static/layout.html'
if os.path.exists(layout_target):
    os.remove(layout_target)
shutil.copy(layout_src, layout_target)

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', '../../../../resource/python_objects.inv'),
}

# Modify regex for sphinx.ext.autosummary.generate.find_autosummary_in_lines.
gfile_abs_path = os.path.abspath(g.__file__)
autosummary_re_line_old = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+autosummary::\s*')"
autosummary_re_line_new = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+(ms[a-z]*)?autosummary::\s*')"
with open(gfile_abs_path, "r+", encoding="utf8") as f:
    data = f.read()
    data = data.replace(autosummary_re_line_old, autosummary_re_line_new)
    exec(data, g.__dict__)

# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
autodoc_source_re = re.compile(r'stringify_signature\(.*?\)')
get_param_func_str = r"""\
import re
import inspect as inspect_

def remove_typehints_content(text):
    # 初始化括号匹配标记，0为无括号包裹
    bracket_count = 0
    start_idx = -1 # 记录第一个":"的位置

    for i, char in enumerate(text):
        # 1. 找到第一个":"，记录起始位置
        if start_idx == -1 and char == ":":
            start_idx = i
            continue

        # 2. 已找到":"，开始判断括号状态
        if start_idx != -1:
            # 遇到"("或"["，括号计数+1（进入括号内）
            if char in ("(", "["):
                bracket_count += 1
            # 遇到")"或"]"，括号计数-1（离开括号内）
            elif char in (")", "]"):
                bracket_count = max(0, bracket_count - 1) # 避免负数值
            # 3. 找到不在括号内的第一个","，执行删除
            elif char == "," and bracket_count == 0:
                return text[:start_idx] + text[i:] # 拼接删除后的内容
            # 4. 找到不在括号内的第一个"="，执行删除
            elif char == "=" and bracket_count == 0:
                return text[:start_idx] + " " +  text[i:] # 拼接删除后的内容，"="前需要有一个空格

    # 若未找到目标","，返回原文本
    return text

def get_param_func(func):
    try:
        source_code = inspect_.getsource(func)
        if func.__doc__:
            source_code = source_code.replace(func.__doc__, '')
        all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
        if "@classmethod" in source_code:
            all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
        else:
            all_params = re.sub("(self)(,|, )?", '', all_params_str[0][0].replace("\n", ""))

        if ":" in all_params:
            colon_idx = all_params.find(":")
            # 处理非最后一个":"以后的内容
            while colon_idx != -1 and "," in all_params[colon_idx+1:]:
                all_params = remove_typehints_content(all_params)
                # 最后一个":"以后的内容中包含","
                if colon_idx == all_params.find(":"):
                    break
                colon_idx = all_params.find(":")

        # 去掉最后一个":"以后的内容
        colon_idx = all_params.find(":")
        if colon_idx != -1:
            # 最后一个":"以后的内容中包含"="，需要保留"="及以后的内容
            if "=" in all_params[colon_idx+1:]:
                all_params = re.sub(":(.*?)=", ' =', all_params)
            # 正常删除最后一个":"以后的内容
            else:
                all_params = re.sub(":.*$", '', all_params)
                # 目前仅有lambda x出现在最后的情况
                if all_params.endswith("lambda x"):
                    all_params += ": ..."
        
        return all_params
    except:
        return ''

def get_obj(obj):
    if isinstance(obj, type):
        return obj.__init__

    return obj
"""

with open(autodoc_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    code_str = autodoc_source_re.sub('"(" + get_param_func(get_obj(self.object)) + ")"', code_str, count=0)
    exec(get_param_func_str, sphinx_autodoc.__dict__)
    exec(code_str, sphinx_autodoc.__dict__)

from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

from sphinx.ext import viewcode
with open('../_ext/overwriteviewcode.txt', 'r', encoding="utf8") as f:
    exec(f.read(), viewcode.__dict__)

with open("../_ext/customdocumenter.txt", "r", encoding="utf8") as f:
    code_str = f.read()
    exec(code_str, sphinx_autodoc.__dict__)

from myautosummary import MsCnAutoSummary

def setup(app):
    app.add_directive('mscnautosummary', MsCnAutoSummary)
    app.add_config_value('rst_files', set(), False)

# Copy source files of chinese python api from golden-stick repository.
from sphinx.util import logging
import shutil
logger = logging.getLogger(__name__)

copy_path = 'docs/api/api_zh_cn'
src_dir_api = os.path.join(os.getenv("GS_PATH"), copy_path)

copy_list = []
moment_dir=os.path.dirname(__file__)

outer_dir = []
for i in os.listdir(src_dir_api):
    if not os.path.isfile(os.path.join(src_dir_api, i)):
        outer_dir.append(os.path.join(src_dir_api, i))

for root, dirs, files in os.walk(src_dir_api):
    root_p = Path(root)
    # ----------- 根目录 -----------
    if root_p == Path(src_dir_api):
        for file in files:
            dst = Path(moment_dir) / file
            if dst.exists():
                os.remove(dst)
            shutil.copy(root_p / file, dst)
            if file.endswith('.rst'):
                content = (root_p / file).read_text(encoding='utf-8')
                if '.. toctree::' in content:
                    continue
                if 'autosummary::' not in content and '\n=====' in content:
                    copy_list.append('./' + file)
        continue

    # ----------- 平铺：仅一级子目录 -----------
    if root_p.parent == Path(src_dir_api):   # 直接子目录
        od_name = root_p.name
        for file in files:
            dst = Path(moment_dir) / od_name / file
            dst.parent.mkdir(exist_ok=True)
            if dst.exists():
                os.remove(dst)
            shutil.copy(root_p / file, dst)
            copy_list.append(str(dst))
        continue

    # ----------- 层级：更深目录 -----------
    rel = '.' + str(root_p).split(copy_path)[-1]
    os.makedirs(rel, exist_ok=True)
    for file in files:
        dst = Path(rel) / file
        if dst.exists():
            os.remove(dst)
        shutil.copy(root_p / file, dst)
        copy_list.append(str(dst))

if os.path.exists(os.path.join(moment_dir, 'index.rst')):
    os.remove(os.path.join(moment_dir, 'index.rst'))
    shutil.copy(os.path.join(os.getenv("GS_PATH"), 'docs/zh_cn/index.rst'),
                os.path.join(moment_dir, 'index.rst'))

gsdocs_image = os.path.join(os.getenv("GS_PATH"), 'docs/zh_cn/images')
if not os.path.exists(os.path.join(moment_dir, 'images')):
    shutil.copytree(gsdocs_image, os.path.join(moment_dir, 'images'))

def extract_toctree(content):
    """
    从index.rst内容中提取所有toctree指令中的文档条目
    """
    entries = re.findall(r'^\s+([a-zA-Z0-9_\-./]+)\s*$', content, re.MULTILINE)
    keywords = ['ptq/', 'quantization/', 'pruner/']
    filtered_entries = [entry for entry in entries if any(keyword in entry for keyword in keywords)]
    spec_copy=[]

    for path in filtered_entries:
        path_split = path.split('/') 
        directory = path_split[0]
        filename = path_split[1]  
        if filename == 'overview':
            gs_p = 'mindspore_gs/' + directory + '/README_CN.md' 
            docs_p = directory + '/' + filename + '.md'
        else:
            if 'round_to_nearest' in filename:
                gs_p = 'mindspore_gs/' + path + '/README_CN.ipynb' 
                docs_p = directory + '/' + filename + '.ipynb'
            else:
                gs_p = 'mindspore_gs/' + path + '/README_CN.md' 
                docs_p = directory + '/' + filename + '.md'
        spec_copy.append([gs_p, docs_p])

    return spec_copy

with open("index.rst", 'r', encoding='utf-8') as f:
    content = f.read()
spec_copy = extract_toctree(content)

for gs_p, f_p in spec_copy:
    ori_p = os.path.join(os.getenv("GS_PATH"), gs_p)
    target_dir = os.path.join(moment_dir, '/'.join(f_p.split('/')[:-1]))
    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(os.path.join(moment_dir, f_p)):
        os.remove(os.path.join(moment_dir, f_p))
    shutil.copy(ori_p, os.path.join(moment_dir, f_p))

    with open(os.path.join(moment_dir, f_p), 'r+', encoding='utf-8') as f:
        content = f.read()
        if f_p.endswith('.md'):
            content = re.sub('.*?/README.md.*\n.*\n', '', content)
        elif f_p.endswith('.ipynb'):
            content = re.sub('\n.*\[View English\].*\n.*\n', '\n', content, 1)
        f.seek(0)
        f.truncate()
        f.write(content)

    images_path = '/'.join(ori_p.split('/')[:-1]) + '/images/zh_cn'
    os.makedirs(os.path.join(target_dir, 'images/zh_cn'), exist_ok=True)
    if os.path.exists(images_path):
        for i in os.listdir(images_path):
            if os.path.exists(os.path.join(target_dir, 'images/zh_cn', i)):
                os.remove(os.path.join(target_dir, 'images/zh_cn', i))
            shutil.copy(os.path.join(images_path, i), os.path.join(target_dir, 'images/zh_cn', i))

# add view
import json

if os.path.exists('../../../../tools/generate_html/version.json'):
    with open('../../../../tools/generate_html/version.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../../tools/generate_html/daily_dev.json'):
    with open('../../../../tools/generate_html/daily_dev.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../../tools/generate_html/daily.json'):
    with open('../../../../tools/generate_html/daily.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)

if os.getenv("GS_PATH").split('/')[-1]:
    copy_repo = os.getenv("GS_PATH").split('/')[-1]
else:
    copy_repo = os.getenv("GS_PATH").split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo.replace('-', '_')][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

re_view = f"\n.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/" + \
          f"resource/_static/logo_source.svg\n    :target: https://gitee.com/mindspore/{copy_repo}/blob/{branch}/"

# master使用
# copy_list白名单转绝对路径（去重）
copy_list_abs = list({Path(p).resolve() for p in copy_list})
# 只遍历白名单文件，添加“查看源文件”链接
inserted = []
for rst_file in copy_list_abs:
    if not rst_file.exists() or not rst_file.suffix == '.rst':
        continue
    try:
        with open(rst_file, 'r+', encoding='utf-8') as f:
            content = f.read()
            new_content = content

            # 跳过自动生成文件
            if '.. include::' in content and '.. automodule::' in content:
                continue

            # 插链接条件：有标题下划线且无 autosummary
            if 'autosummary::' not in content and "\n=====" in content:
                rel_path = rst_file.relative_to(Path(moment_dir)).as_posix()
                re_view_ = re_view + copy_path + '/' + rel_path + '\n    :alt: 查看源文件\n\n'
                new_content = re.sub(r'([=]{5,})\n', r'\1\n' + re_view_, content, 1)
            if new_content != content:
                    f.seek(0)
                    f.truncate()
                    f.write(new_content)
    except Exception:
        print(f'打开{i}文件失败')

if not os.path.exists(os.path.join(moment_dir, 'install.md')):
    shutil.copy(os.path.join(os.getenv("GS_PATH"), 'docs/zh_cn/install.md'),
                os.path.join(moment_dir, 'install.md'))
    with open(os.path.join(moment_dir, 'install.md'), 'r+', encoding='utf-8') as f:
        content = f.read()
        content = re.sub('\n\[View English\].*\n', '', content, 1)
        f.seek(0)
        f.truncate()
        f.write(content)

if not os.path.exists(os.path.join(moment_dir, 'design.md')):
    shutil.copy(os.path.join(os.getenv("GS_PATH"), 'docs/zh_cn/design.md'),
                os.path.join(moment_dir, 'design.md'))
    with open(os.path.join(moment_dir, 'design.md'), 'r+', encoding='utf-8') as f:
        content = f.read()
        content = re.sub('\n\[View English\].*\n', '', content, 1)
        f.seek(0)
        f.truncate()
        f.write(content)

if not os.path.exists(os.path.join(moment_dir, 'CONTRIBUTING.md')):
    shutil.copy(os.path.join(os.getenv("GS_PATH"), 'CONTRIBUTING_CN.md'),
                os.path.join(moment_dir, 'CONTRIBUTING.md'))
    with open(os.path.join(moment_dir, 'CONTRIBUTING.md'), 'r+', encoding='utf-8') as f:
        content = f.read()
        content = re.sub('\n\[View English\].*\n', '', content, 1)
        f.seek(0)
        f.truncate()
        f.write(content)

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
import nbsphinx_mod

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

src_release = os.path.join(os.getenv("GS_PATH"), 'RELEASE_CN.md')
des_release = "./RELEASE.md"
with open(src_release, "r", encoding="utf-8") as f:
    data = f.read()
if len(re.findall("\n## (.*?)\n",data)) > 1:
    content = re.findall("(## [\s\S\n]*?)\n## ", data)
else:
    content = re.findall("(## [\s\S\n]*)", data)
#result = content[0].replace('# MindSpore', '#', 1)
with open(des_release, "w", encoding="utf-8") as p:
    p.write("# Release Notes"+"\n\n")
    p.write(content[0])

# 发版本时这里启用
# re_url = r"(((gitee.com/mindspore/docs)|(github.com/mindspore-ai/(mindspore|docs))|" + \
#          r"(mindspore.cn/(docs|tutorials|lite))|(obs.dualstack.cn-north-4.myhuaweicloud)|" + \
#          r"(mindspore-website.obs.cn-north-4.myhuaweicloud))[\w\d/_.-]*?)/(master)"

# re_url2 = r"(gitee.com/mindspore/(mindspore|mindspore-lite)/[\w\d/_.-]*?)/(master)"

# re_url3 = r"(((gitee.com/mindspore/golden-stick)|(mindspore.cn/golden_stick))/[\w\d/_.-]*?)/(master)"

# re_url4 = r"(mindspore.cn/vllm_mindspore/[\w\d/_.-]*?)/(master)"

# re_url5 = r"(((gitee.com/mindspore/mindformers)|(mindspore.cn/mindformers))[\w\d/_.-]*?)/(dev)"

# 发版本时这里启用
# for cur, _, files in os.walk(moment_dir):
#     for i in files:
#         if i.endswith('.rst') or i.endswith('.md') or i.endswith('.ipynb'):
#             try:
#                 with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
#                     content = f.read()
#                     new_content = re.sub(re_url, r'\1/r2.7.1', content)
#                     new_content = re.sub(re_url2, r'\1/v2.7.1', new_content)
#                     new_content = re.sub(re_url3, r'\1/r1.3.0', new_content)
#                     new_content = re.sub(re_url4, r'\1/r0.4.0', new_content)
#                     new_content = re.sub(re_url5, r'\1/r1.7.0', new_content)
#                     if new_content != content:
#                         f.seek(0)
#                         f.truncate()
#                         f.write(new_content)

#             except Exception:
#                 print(f'打开{i}文件失败')