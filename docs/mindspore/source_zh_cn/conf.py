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

from genericpath import exists
import re
import os
import sys
import glob
import IPython
import shutil
import regex
import sphinx

# Fix some dl-label lack class='simple'
from docutils.writers import _html_base

with open(_html_base.__file__, "r+", encoding="utf-8") as f:
    code_str = f.read()
    old_str = '''        if self.is_compactable(node):
            classes.append('simple')'''
    new_str = '''        if classes == []:
            classes.append('simple')'''
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, _html_base.__dict__)

from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

from sphinx.ext import viewcode
with open('../_ext/overwriteviewcode.txt', 'r', encoding="utf8") as f:
    exec(f.read(), viewcode.__dict__)

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

with open(autodoc_source_path, "r+", encoding="utf8") as f:
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
    'sphinx.ext.viewcode',
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

html_search_options = {'dict': '../../resource/jieba.txt'}

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
with open(gfile_abs_path, "r+", encoding="utf8") as f:
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
    if not exists(sourcedir):
        logger.warning(f"不存在目录：{sourcedir}！")
    if os.path.exists(des_sir):
        shutil.rmtree(des_sir)
    shutil.copytree(sourcedir, des_sir)

copy_source(src_dir, des_sir)

probability_dir = './api_python/probability'
if os.path.exists(probability_dir):
    shutil.rmtree(probability_dir)

# 删除并获取ops下多余的接口文件名
white_list = ['mindspore.ops.comm_note.rst']

ops_adjust = []

refer_ops_adjust = []

func_adjust = []

def ops_interface_name():
    dir_list = ['mindspore.ops.primitive.rst', 'mindspore.ops.rst']
    interface_name_list = []
    all_rst = []

    for i in dir_list:
        target_path = os.path.join(src_dir, i)
        with open(target_path,'r+',encoding='utf8') as f:
            content =  f.read()
        new_content = content
        if 'primitive' in i:
            for name in ops_adjust:
                new_content = new_content.replace('    mindspore.ops.' + name + '\n', '')
            for name in refer_ops_adjust:
                new_content = new_content.replace('    mindspore.ops.' + name + '\n', '')
            primi_list = re.findall("    (mindspore\.ops\.\w*?)\n", new_content)
            if not os.path.exists(os.path.join(os.path.dirname(__file__), 'api_python', i)):
                return primi_list
        else:
            for name in func_adjust:
                new_content = new_content.replace('    mindspore.ops.' + name + '\n', '')
        if interface_name_list:
            interface_name_list = interface_name_list + re.findall("(mindspore\.ops\.\w*)", new_content)
        else:
            interface_name_list = re.findall("(mindspore\.ops\.\w*)", new_content)

        if new_content != content and os.path.exists(os.path.join(os.path.dirname(__file__), 'api_python', i)):
            with open(os.path.join(os.path.dirname(__file__), 'api_python', i),'r+',encoding='utf8') as g:

                g.seek(0)
                g.truncate()
                g.write(new_content)

    for j in os.listdir(os.path.join(src_dir, 'ops')):
        if j.split('.')[-1]=='rst' and 'ops.silent_check.' not in j:
            all_rst.append(j.replace('.rst', '').replace("func_", ''))

    extra_interface_name = set(all_rst).difference(set(interface_name_list))
    print(extra_interface_name)
    if extra_interface_name:
        with open(os.path.join(os.path.dirname(__file__),'extra_interface_del.txt'),'w+',encoding='utf8') as g:
            extra_write_list = []
            for k in extra_interface_name:
                extra_file = k +'.rst'
                if os.path.exists(os.path.join(os.path.dirname(__file__), 'api_python/ops', extra_file)) and extra_file not in white_list:
                    os.remove(os.path.join(os.path.dirname(__file__), 'api_python/ops', extra_file))
                    extra_write_list.append(extra_file)
            g.write(str(extra_write_list))

    return primi_list

# 删除并获取nn下多余的接口文件名
def nn_interface_name():
    interface_name_list = []
    target_path = os.path.join(os.path.dirname(__file__),'api_python','mindspore.nn.rst')
    with open(target_path,'r+',encoding='utf8') as f:
        content =  f.read()
        interface_name_list = re.findall("mindspore\.nn\.(\w*)",content)
    all_rst = []
    for j in os.listdir(os.path.join(os.path.dirname(__file__),'api_python/nn')):
        if j.split('.')[-1]=='rst':
            if 'optim_' not in j:
                all_rst.append(j.split('.')[-2])

    extra_interface_name = set(all_rst).difference(set(interface_name_list))
    print(extra_interface_name)
    if extra_interface_name:
        with open(os.path.join(os.path.dirname(__file__),'extra_interface_del.txt'),'a+',encoding='utf8') as g:
            extra_write_list = []
            for k in extra_interface_name:
                k = "mindspore.nn." + k +'.rst'
                if os.path.exists(os.path.join(os.path.dirname(__file__),'api_python/nn',k)):
                    os.remove(os.path.join(os.path.dirname(__file__),'api_python/nn',k))
                    extra_write_list.append(k)
            g.write(str(extra_write_list))

# 删除并获取Tensor下多余的接口文件名
def tensor_interface_name():
    interface_name_list = []
    target_path = os.path.join(os.path.dirname(__file__),'api_python/mindspore','mindspore.Tensor.rst')
    with open(target_path,'r+',encoding='utf8') as f:
        content =  f.read()
        interface_name_list = re.findall("mindspore\.Tensor\.(\w*)",content)
    all_rst = []
    for j in os.listdir(os.path.join(os.path.dirname(__file__),'api_python/mindspore/Tensor')):
        if j.split('.')[-1]=='rst':
            all_rst.append(j.split('.')[-2])

    extra_interface_name = set(all_rst).difference(set(interface_name_list))
    print(extra_interface_name)
    if extra_interface_name:
        with open(os.path.join(os.path.dirname(__file__),'extra_interface_del.txt'),'a+',encoding='utf8') as g:
            extra_write_list = []
            for k in extra_interface_name:
                k = "mindspore.Tensor." + k +'.rst'
                if os.path.exists(os.path.join(os.path.dirname(__file__),'api_python/mindspore/Tensor',k)):
                    os.remove(os.path.join(os.path.dirname(__file__),'api_python/mindspore/Tensor',k))
                    extra_write_list.append(k)
            g.write(str(extra_write_list))

# Rename .rst file to .txt file for include directive.
from rename_include import rename_include

rename_include('api_python')

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
        if i.endswith('.rst') or i.endswith('.md'):
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
target_dir_ops="./api_python/ops/"
try:
    for filename in os.listdir(target_dir_ops):
        newname = filename.replace("func_",'')
        os.rename(os.path.join(target_dir_ops, filename),os.path.join(target_dir_ops, newname))
except Exception as e:
    print(e)

target_dir_mindspore="./api_python/mindspore/"
try:
    for filename in os.listdir(target_dir_mindspore):
        newname = filename.replace("func_",'')
        os.rename(os.path.join(target_dir_mindspore, filename),os.path.join(target_dir_mindspore, newname))
except Exception as e:
    print(e)

target_dir_tensor="./api_python/mindspore/Tensor/"
try:
    for filename in os.listdir(target_dir_tensor):
        newname = filename.replace("method_",'')
        os.rename(os.path.join(target_dir_tensor, filename),os.path.join(target_dir_tensor, newname))
except Exception as e:
    print(e)

primitive_list = ops_interface_name()

try:
    nn_interface_name()
    tensor_interface_name()
except Exception as e:
    print(e)

# auto generate rst by en
from generate_rst_by_en import generate_rst_by_en

exist_rst_file, primi_auto = generate_rst_by_en(primitive_list, './api_python/ops')
if exist_rst_file:
    print(f'自动生成 ops API 中文时被覆盖的rst文件如下：\n{exist_rst_file}')

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary, MsCnAutoSummary, MsCnPlatformAutoSummary, MsCnNoteAutoSummary, MsCnPlatWarnAutoSummary

rst_files = set([i.replace('.rst', '') for i in glob.glob('api_python/**/*.rst', recursive=True)])

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)
    app.add_directive('mscnautosummary', MsCnAutoSummary)
    app.add_directive('mscnplatformautosummary', MsCnPlatformAutoSummary)
    app.add_directive('mscnplatwarnautosummary', MsCnPlatWarnAutoSummary)
    app.add_directive('mscnnoteautosummary', MsCnNoteAutoSummary)
    app.add_config_value('rst_files', set(), False)
    app.add_directive('includecode', IncludeCodeDirective)
    app.add_js_file('js/mermaid-9.3.0.js')


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

src_release = os.path.join(repo_path, 'RELEASE_CN.md')
des_release = "./RELEASE.md"
release_source = f'[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/resource/_static/logo_source.svg)](https://gitee.com/mindspore/{copy_repo}/blob/{branch}/' + 'RELEASE_CN.md)\n'

with open(src_release, "r", encoding="utf-8") as f:
    data = f.read()
if len(re.findall("\n## (.*?)\n",data)) > 1:
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
