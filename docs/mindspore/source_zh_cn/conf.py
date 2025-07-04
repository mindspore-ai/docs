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

import re
import os
import sys
import glob
import IPython
import shutil
import inspect
import regex
import sphinx

# Fix some dl-label lack class='simple'
from docutils.writers import _html_base

with open(_html_base.__file__, "r", encoding="utf-8") as f:
    code_str = f.read()
    old_str = '''        if self.is_compactable(node):
            classes.append('simple')'''
    new_str = '''        if classes == []:
            classes.append('simple')'''
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, _html_base.__dict__)

# 提取样例、支持平台至中文
from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

# 调整 Tensor 重载函数的定义名称显示
from sphinx.domains import python as domain_py
with open(domain_py.__file__, 'r', encoding="utf8") as f:
    code_str = f.read()
    old_str = "signode += addnodes.desc_addname(nodetext, nodetext)"
    new_str = """signode += addnodes.desc_addname(nodetext, nodetext)
        elif 'mindspore.Tensor' in classname:
            signode += addnodes.desc_addname('mindspore.Tensor.', 'mindspore.Tensor.')
        elif 'mindspore.mint' in classname:
            signode += addnodes.desc_addname('mindspore.mint.', 'mindspore.mint.')"""
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, domain_py.__dict__)

# # 添加源代码链接
# from sphinx.ext import viewcode
# with open('../_ext/overwriteviewcode.txt', 'r', encoding="utf8") as f:
#     exec(f.read(), viewcode.__dict__)

from sphinx.ext import linkcode
with open('../_ext/overwritelinkcode.txt', 'r', encoding="utf8") as f:
    exec(f.read(), linkcode.__dict__)

from docutils import statemachine

with open(statemachine.__file__, 'r') as g:
    code = g.read().replace("assert len(self.data) == len(self.items), 'data mismatch'", "#assert len(self.data) == len(self.items), 'data mismatch'")
    exec(code, statemachine.__dict__)

sys.path.append(os.path.abspath('../_ext'))
import sphinx.ext.autosummary.generate as g

from sphinx.ext import autodoc as sphinx_autodoc
# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
autodoc_source_re = re.compile(r'stringify_signature\(.*?\)')
get_param_func_str = r"""\
import re
import inspect as inspect_

def get_param_func(func):
    try:
        source_code = inspect_.getsource(func)
        if func.__doc__:
            source_code = source_code.replace(func.__doc__, '')
        all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
        all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
        return all_params
    except:
        return ''

def get_obj(obj):
    if isinstance(obj, type):
        return obj.__init__

    return obj
"""

with open(autodoc_source_path, "r", encoding="utf8") as f:
    code_str = f.read()
    code_str = autodoc_source_re.sub('"(" + get_param_func(get_obj(self.object)) + ")"', code_str, count=0)
    exec(get_param_func_str, sphinx_autodoc.__dict__)
    exec(code_str, sphinx_autodoc.__dict__)

with open("../_ext/customdocumenter.txt", "r", encoding="utf8") as f:
    code_str = f.read()
    exec(code_str, sphinx_autodoc.__dict__)

# Fix mathjax tags
from sphinx.ext import mathjax as sphinx_mathjax

with open(sphinx_mathjax.__file__, "r", encoding="utf-8") as f:
    code_str = f.read()
    old_str = r'''        if r'\\' in part:
            self.body.append(r'\begin{split}' + part + r'\end{split}')'''
    new_str = r'''        if r'\\' in part:
            if r'\tag{' in part:
                part1, part2 = part.split(r'\tag{')
                self.body.append(r'\begin{split}' + part1 + r'\end{split}' + r'\tag{' +part2)
            else:
                self.body.append(r'\begin{split}' + part + r'\end{split}')'''
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, sphinx_mathjax.__dict__)

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = 'MindSpore'
author = 'MindSpore'
# language = 'cn'
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
    'sphinx.ext.linkcode',
    'sphinxcontrib.mermaid',
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

# locale_dirs = ['locale/']

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

html_static_path = ['_static']

mermaid_version = ""

mermaid_init_js = ""

# -- Options for HTML output -------------------------------------------------

# Reconstruction of sphinx auto generated document translation.
language = 'zh_CN'
locale_dirs = ['../../../resource/locale/']
gettext_compact = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

import sphinx_rtd_theme
layout_target = os.path.join(os.path.dirname(sphinx_rtd_theme.__file__), 'layout.html')
layout_src = '../../../resource/_static/layout.html'
if os.path.exists(layout_target):
    os.remove(layout_target)
shutil.copy(layout_src, layout_target)

html_search_language = 'zh'

import jieba

jieba.load_userdict('../../../resource/jieba.txt')

sys.path.append(os.path.abspath('../../../resource/sphinx_ext'))
# import anchor_mod
import nbsphinx_mod

sys.path.append(os.path.abspath('../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective

# -- Options for Texinfo output -------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../resource/numpy_objects.inv'),
}

# Modify regex for sphinx.ext.autosummary.generate.find_autosummary_in_lines.
gfile_abs_path = os.path.abspath(g.__file__)
autosummary_re_line_old = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+autosummary::\s*')"
autosummary_re_line_new = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+(ms[a-z]*)?autosummary::\s*')"
with open(gfile_abs_path, "r", encoding="utf8") as f:
    data = f.read()
    data = data.replace(autosummary_re_line_old, autosummary_re_line_new)
    exec(data, g.__dict__)


sys.path.append(os.path.abspath('../../../resource/search'))
import search_code

# Copy source files of chinese python api from mindspore repository.
from sphinx.util import logging
logger = logging.getLogger(__name__)

copy_path = 'docs/api/api_python'
repo_path = os.getenv("MS_PATH")
src_dir = os.path.join(repo_path, copy_path)
des_sir = "./api_python"

def copy_source(sourcedir, des_sir):
    for i in os.listdir(sourcedir):
        if os.path.isfile(os.path.join(sourcedir,i)):
            if os.path.exists(os.path.join(des_sir, i)):
                os.remove(os.path.join(des_sir, i))
            shutil.copy(os.path.join(sourcedir, i), os.path.join(des_sir, i))
        else:
            if os.path.exists(os.path.join(des_sir, i)):
                shutil.rmtree(os.path.join(des_sir, i))
            shutil.copytree(os.path.join(sourcedir, i), os.path.join(des_sir, i))

no_viewsource_list = [os.path.join(des_sir,i) for i in os.listdir(des_sir)]

copy_source(src_dir, des_sir)

probability_dir = './api_python/probability'
if os.path.exists(probability_dir):
    shutil.rmtree(probability_dir)

# 删除多余的接口的文件
white_list = ['mindspore.ops.comm_note.rst', 'mindspore.mint.comm_note.rst']

refer_ops_adjust = []

primi_del = []
ops_del = []
nn_del = []
mint_del = []
numpy_del = []
scipy_del = []
tensor_del = []

def del_redundant_api_file(base_path, f_path, d_path, re_str, del_list, spec_str=''):
    interface_name_list = []
    all_dir_rst = []
    d_path = os.path.join(base_path, d_path)
    for fp in f_path:
        fp_ = os.path.join(base_path, fp)
        if not os.path.exists(fp_):
            print(f'{fp_}不存在！')
            return False
        with open(fp_, 'r+', encoding='utf8') as g:
            content =  g.read()
            new_content = content
            for dn in del_list:
                new_content = re.sub(r'[ ]{3,4}'+dn+'\n', '', new_content)
            if interface_name_list:
                interface_name_list = interface_name_list + re.findall('[ ]{3,4}('+re_str+'.+)\n', new_content)
            else:
                interface_name_list = re.findall('[ ]{3,4}('+re_str+'.+)\n', new_content)

            if new_content != content:
                g.seek(0)
                g.truncate()
                g.write(new_content)
    for j in os.listdir(d_path):
        if j.split('.')[-1]=='rst':
            if spec_str and spec_str in j:
                continue
            else:
                all_dir_rst.append(j.replace('.rst', '').replace("func_", '').replace("method_", ''))

    extra_interface_name = set(all_dir_rst).difference(set(interface_name_list))

    print(f'冗余文件的接口名如下:\n{extra_interface_name}')
    for k in extra_interface_name:
        extra_file = k +'.rst'
        if os.path.exists(os.path.join(d_path, extra_file)) and extra_file not in white_list:
            os.remove(os.path.join(d_path, extra_file))
    return True

# Convert encoding for api files.
import chardet
import codecs

api_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_python')

def convert2utf8(filename):
    f = codecs.open(filename, 'rb')
    content = f.read()
    source_encoding = chardet.detect(content)['encoding']
    if source_encoding == None:
        pass
    elif source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
        content = content.decode(source_encoding, 'ignore')
        codecs.open(filename, 'w', encoding='UTF-8-SIG').write(content)
    f.close()

for root, dirs, files in os.walk(api_file_dir, topdown=True):
    for file_ in files:
        if '.rst' in file_ or '.txt' in file_:
            convert2utf8(os.path.join(root, file_))

# Rename .rst file to .txt file for include directive.
from rename_include import rename_include

rename_include('api_python')
rename_include('migration_guide')

# modify urls
import json

if os.path.exists('../../../tools/generate_html/version.json'):
    with open('../../../tools/generate_html/version.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../tools/generate_html/daily_dev.json'):
    with open('../../../tools/generate_html/daily_dev.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../tools/generate_html/daily.json'):
    with open('../../../tools/generate_html/daily.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)

if repo_path.split('/')[-1]:
    copy_repo = repo_path.split('/')[-1]
else:
    copy_repo = repo_path.split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

re_view = f"\n.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/" + \
          f"resource/_static/logo_source.svg\n    :target: https://gitee.com/mindspore/{copy_repo}/blob/{branch}/"

for cur, _, files in os.walk(des_sir):
    for i in files:
        if os.path.join(cur, i) in no_viewsource_list:
            continue
        if i.endswith('.rst') or i.endswith('.md') or i.endswith('.ipynb'):
            try:
                with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
                    content = f.read()
                    new_content = content
                    if i.endswith('.md'):
                        md_view = f'[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/resource/_static/logo_source.svg)](https://gitee.com/mindspore/{copy_repo}/blob/{branch}/' + copy_path + cur.split('api_python')[-1] + '/' + i + ')\n\n'
                        if 'resource/_static/logo_source' not in new_content:
                            new_content = re.sub('(# .*\n\n)', r'\1'+ md_view, new_content, 1)
                    if new_content != content:
                        f.seek(0)
                        f.truncate()
                        f.write(new_content)
            except Exception:
                print(f'打开{i}文件失败')
        if i.endswith('.rst'):
            try:
                with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
                    content = f.read()
                    new_content = content
                    if '.. include::' in content and '.. automodule::' in content:
                        continue
                    if 'autosummary::' not in content and "\n=====" in content:
                        re_view_ = re_view + copy_path + cur.split('api_python')[-1] + '/' + i +'\n    :alt: 查看源文件\n\n'
                        new_content = re.sub('([=]{5,})\n', r'\1\n' + re_view_, content, 1)
                    if new_content != content:
                        f.seek(0)
                        f.truncate()
                        f.write(new_content)
            except Exception:
                print(f'打开{i}文件失败')

# rename file name to solve Case sensitive.

rename_list = [("./api_python/ops/", "func_", ""),
               ("./api_python/mindspore/", "func_", ""),
               ("./api_python/mindspore/Tensor/", "method_", ""),
               ("./api_python/mint/", "func_", "")]

try:
    for tp in rename_list:
        for filename in os.listdir(tp[0]):
            newname = filename.replace(tp[1], tp[2])
            os.rename(os.path.join(tp[0], filename),os.path.join(tp[0], newname))
except Exception as e:
    print(e)

del_redundant_api_file(des_sir, ['mindspore.ops.rst', 'mindspore.ops.primitive.rst'], f'ops', 'mindspore.ops.', ops_del+primi_del, 'ops.silent_check.')
del_redundant_api_file(des_sir, ['mindspore.nn.rst'], 'nn', 'mindspore.nn.', nn_del, 'optim_')
del_redundant_api_file(des_sir, ['mindspore.mint.rst'], 'mint', 'mindspore.mint.', mint_del)
del_redundant_api_file(des_sir, ['mindspore.numpy.rst'], 'numpy', 'mindspore.numpy.', numpy_del)
del_redundant_api_file(des_sir, ['mindspore.scipy.rst'], 'scipy', 'mindspore.scipy.', scipy_del)
del_redundant_api_file(des_sir, ['mindspore/mindspore.Tensor.rst'], 'mindspore/Tensor', 'mindspore.Tensor.', tensor_del)

# auto generate rst by en
from generate_rst_by_en import generate_rst_by_en

with open(os.path.join(src_dir, 'mindspore.ops.primitive.rst'), 'r', encoding='utf-8') as f:
    content = f.read()
for i in primi_del:
    content = re.sub(r'[ ]{3,4}'+i+'\n', '', content)
primitive_sum = re.findall(r"[ ]{3,4}(mindspore\.ops\..+)\n", content)

exist_rst_file, primi_auto = generate_rst_by_en(primitive_sum, './api_python/ops')
if exist_rst_file:
    print(f'自动生成 ops API 中文时被覆盖的rst文件如下：\n{exist_rst_file}')

with open(os.path.join(src_dir, 'mindspore.mint.rst'), 'r', encoding='utf-8') as f:
    content = f.read()
for i in mint_del:
    content = re.sub(r'[ ]{3,4}'+i+'\n', '', content)
mint_sum = re.findall(r"[ ]{3,4}(mindspore\.mint\..+)\n", content)

exist_rst_file, mint_auto = generate_rst_by_en(mint_sum, './api_python/mint')
if exist_rst_file:
    print(f'自动生成 mint API 中文的rst文件如下：\n{exist_rst_file}')

# auto generate rst for mint from ops

from generate_ops_mint_rst import generate_ops_mint_rst

try:
    generate_ops_mint_rst(repo_path, os.path.join(src_dir, 'ops'), "./api_python/mint")
except Exception as e:
    print(e)

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary, MsCnAutoSummary, MsCnPlatformAutoSummary, MsCnNoteAutoSummary, MsCnPlatWarnAutoSummary, MsCnPlataclnnAutoSummary

rst_files = set([i.replace('.rst', '') for i in glob.glob('api_python/**/*.rst', recursive=True)])

# 获取mint和aclnn算子的映射关系
mint_aclnn_path = '../../../resource/api_updates/mint_aclnn_cn.md'
with open(mint_aclnn_path, 'r', encoding='utf-8') as f:
    mint_aclnn_content = f.read()

mint_aclnn = dict()
for mint_n, aclnn_str in re.findall(r'\n\| \[(.*?)\].*?\|(.*?)\|', mint_aclnn_content):
    new_mint_n = 'mindspore.' + mint_n
    all_aclnn = ''
    for aclnn_n, url in re.findall(r'\[(aclnn.*?)\]\((.*?)\)', aclnn_str):
        all_aclnn += f'`{aclnn_n} <{url}>`_ , '
    if not all_aclnn:
        all_aclnn = '无'
    mint_aclnn[new_mint_n] = all_aclnn

repo_whl = 'mindspore/python/'
giturl = 'https://gitee.com/mindspore/'
ops_yaml = 'mindspore/ops/op_def/yaml/doc/'
tensor_yaml = 'mindspore/ops/api_def/method_doc/'
func_yaml = 'mindspore/ops/api_def/function_doc/'

try:
    ops_yaml_list = [i for i in os.listdir(os.path.join(repo_path, 'mindspore/ops/op_def/yaml/doc')) if i.endswith('_doc.yaml') and '_grad' not in i]
except:
    ops_yaml_list = []

try:
    tensor_yaml_list = [i for i in os.listdir(os.path.join(repo_path, 'mindspore/ops/api_def/method_doc/')) if i.endswith('_doc.yaml') and '_grad' not in i]
except:
    tensor_yaml_list = []

func_name_dict = {}
for i in os.listdir(os.path.join(repo_path, 'mindspore/ops/op_def/yaml')):
    if i.endswith('_op.yaml') and '_grad' not in i:
        with open(os.path.join(repo_path, 'mindspore/ops/op_def/yaml', i), 'r+', encoding='utf-8') as f:
            op_content = f.read()
            if re.findall('function:\n\s+?name: (.*)', op_content):
                func_name_dict[re.findall('function:\n\s+?name: (.*)', op_content)[0]] = i.replace('_op.yaml', '')

import mindspore

# Use the linkcode extension to override [SOURCE] links to point to the gitee repo.
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    name = info["fullname"]
    modname = info['module']
    if name and not info['module']:
        if name.split('.')[-1].lower() == name.split('.')[-1] and name.split('.')[-2].lower() != name.split('.')[-2]:
            modname = '.'.join(name.split('.')[:-2])
            name = '.'.join(name.split('.')[-2:])
        else:
            modname = '.'.join(name.split('.')[:-1])
            name = name.split('.')[-1]
    try:
        module = __import__(modname, fromlist=[''])
        obj = module
        for part in name.split("."):
            obj = getattr(obj, part)
        # Get the source file and line number
        obj = inspect.unwrap(obj)
        pkg_fn = inspect.getsourcefile(obj)
        # 自动生成的ops模块单独处理
        py_source_rel = ''
        if 'mindspore/ops/auto_generate/' in pkg_fn:
            name1 = name
            spec_tp = [('mint.nn.functional.dense', 'mint.nn.functional.linear', 'dense', 'linear'),
                       ('mint.select_ext_view', 'mint.select', 'select_ext_view', 'select'),
                       ('mint.transpose_ext_view', 'mint.transpose', 'transpose_ext_view', 'transpose'),
                       ('mint.nn.functional.im2col_ext', 'mint.nn.functional.unfold', 'im2col_ext', 'unfold'),
                       ('mint.nn.functional.inplace_threshold', 'mint.nn.functional.threshold_', 'inplace_threshold', 'threshold_'),
                       ]
            fullname = modname + '.' + name
            for i in spec_tp:
                if fullname.endswith(i[1]):
                    name1 = name.replace(i[3], i[2])
            # 根据接口名内大写字母个数分类处理primitive，得到yaml文件名
            if fullname not in primi_auto:
                if len(re.findall('[A-Z]', name1)) == 1:
                    name1 = name1.lower()
                elif len(re.findall('[A-Z]', name1)) > 1:
                    name1 = 'mindspore.ops.' + '_'.join(re.split('(?=[A-Z])', name1)[1:]).lower()
                    if name1.endswith('_d'):
                        name1 = name1[:-2] + 'd'
                    if name1.split('.')[-1] + '_doc.yaml' not in ops_yaml_list:
                        if name.split('.')[-1].lower() + '_doc.yaml' in ops_yaml_list:
                            name1 = name.lower()
            else:
                return None
            # 根据yaml文件名查询文件是否存在，分别再处理
            if name1.split('.')[-1] + '_ext_doc.yaml' in ops_yaml_list and '.mint.' in fullname:
                py_source_rel = ops_yaml + name1.split('.')[-1] + '_ext_doc.yaml'
            elif name1.split('.')[-1] + '_doc.yaml' not in ops_yaml_list:
                # 新增查找_ext后缀文件
                if name1.split('.')[-1] + '_ext_doc.yaml' in ops_yaml_list:
                    py_source_rel = ops_yaml + name1.split('.')[-1] + '_ext_doc.yaml'
                else:
                    for f_yaml in ops_yaml_list:
                        # 对文件名中存在v[0-9]的特殊处理
                        if re.findall(f"{name1.split('.')[-1]}_v[0-9]+_doc.yaml", f_yaml):
                            py_source_rel = ops_yaml + re.findall(f"{name1.split('.')[-1]}_v[0-9]+_doc.yaml", f_yaml)[0]
                            break
                    else:
                        py_source_rel = ''
            else:
                py_source_rel = ops_yaml + name1.split('.')[-1] + '_doc.yaml'

            if name1.split('.')[-1] in func_name_dict and not py_source_rel:
                py_source_rel = ops_yaml + func_name_dict[name1.split('.')[-1]] + '_doc.yaml'
        elif 'ops/functional_overload' in pkg_fn:
            py_source_rel = func_yaml + name.split('.')[-1] + '_doc.yaml'

        if py_source_rel:
            return f"https://gitee.com/mindspore/mindspore/blob/{branch}/{py_source_rel}"
        elif 'mindspore/ops/auto_generate/' in py_source_rel:
            return None
        source, linenum = inspect.getsourcelines(obj)
    except Exception:
        name = info["fullname"]
        if name.startswith('mindspore.Tensor.') and name.split('.')[-1] + '_doc.yaml' in tensor_yaml_list:
            py_source_rel = tensor_yaml + name.split('.')[-1] + '_doc.yaml'
            return f"https://gitee.com/mindspore/mindspore/blob/{branch}/{py_source_rel}"
        return None

    pkg_fn = os.path.relpath(pkg_fn, start=os.path.dirname(mindspore.__file__))
    return f"https://gitee.com/mindspore/mindspore/blob/{branch}/{repo_whl}{copy_repo}/{pkg_fn}#L{linenum}"

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)
    app.add_directive('mscnautosummary', MsCnAutoSummary)
    app.add_directive('mscnplatformautosummary', MsCnPlatformAutoSummary)
    app.add_directive('mscnplatwarnautosummary', MsCnPlatWarnAutoSummary)
    app.add_directive('mscnplataclnnautosummary', MsCnPlataclnnAutoSummary)
    app.add_directive('mscnnoteautosummary', MsCnNoteAutoSummary)
    app.add_config_value('rst_files', set(), False)
    app.add_config_value('mint_aclnn', {}, True)
    app.add_directive('includecode', IncludeCodeDirective)
    app.add_js_file('js/mermaid-9.3.0.js')

src_release = os.path.join(repo_path, 'RELEASE_CN.md')
des_release = "./RELEASE.md"
release_source = f'[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/resource/_static/logo_source.svg)](https://gitee.com/mindspore/{copy_repo}/blob/{branch}/' + 'RELEASE_CN.md)\n'

with open(src_release, "r", encoding="utf-8") as f:
    data = f.read()

hide_release = []
if len(re.findall("\n## (.*?)\n",data)) > 1:
    for i in hide_release:
        del_doc = re.findall(f"(\n## MindSpore {i}[\s\S\n]*?)\n## ", data)
        if del_doc:
            data = data.replace(del_doc[0], '')
    content = regex.findall("(\n## MindSpore [^L][\s\S\n]*?)\n## ", data, overlapped=True)
    repo_version = re.findall("\n## MindSpore ([0-9]+?\.[0-9]+?)\.([0-9]+?)[ -]", content[0])[0]
    content_new = ''
    for i in content:
        if re.findall(f"\n## MindSpore ({repo_version[0]}\.[0-9]+?)[ -]", i):
            content_new += i
    content = content_new
else:
    content = re.findall("(\n## [\s\S\n]*)", data)
    content = content[0]

with open(des_release, "w", encoding="utf-8") as p:
    p.write("# Release Notes" + "\n\n" + release_source)
    p.write(content)
