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
import sphinx

# Get side effect class name and add it to notes.
from sphinx.util import logging

logger = logging.getLogger(__name__)
api_en_dir_nn = os.path.join(os.getenv("MS_PATH"), 'mindspore/python/mindspore/nn')
api_en_dir_ops = os.path.join(os.getenv("MS_PATH"), 'mindspore/python/mindspore/ops')
target_api_en_dir = [api_en_dir_nn, api_en_dir_ops]
py_file_list = []
for j in target_api_en_dir:
    for root,dirs,files in os.walk(j):
        for i in files:
            if re.findall("^.*\.py$",i):
                py_file_path = os.path.join(root,i)
                with open(py_file_path,'r+',encoding='utf8') as f:
                    content = f.read()
                    if re.findall("add_prim_attr\((\'|\")side_effect_(mem|io)", content):
                        py_file_list.append(py_file_path)

for i in py_file_list:
    with open(i,'r+',encoding='utf8') as f:
        side_effect_str_list = []
        position_count = dict()
        content = f.read()
        side_effect_type = re.findall(r"add_prim_attr\((\'|\")side_effect_(mem|io)(.*?)\)", content)
        for k in side_effect_type:
            side_effect_str = "add_prim_attr(" + k[0] + "side_effect_" + k[1] + k[2] + ')'
            if 'True' in k[2]:
                if side_effect_str not in side_effect_str_list:
                    side_effect_str_list.append(side_effect_str)
                    position_count[side_effect_str] = 1
                else:
                    position_count[side_effect_str] += 1
        index_side_effect = 0  
        for key,values in position_count.items():
            for z in range(values):
                if z==0:
                    index_side_effect = content.find(key)
                else:
                    index_side_effect = content.find(key, index_side_effect+1)
                cls_name = re.findall(r"class (.*?):\n.*?\"\"\"", content[:index_side_effect])[-1]
                if content.rfind('    Supported Platforms:', 0, index_side_effect)>content.find(cls_name):
                    str_notes = content[content.find(cls_name):content.rfind('    Supported Platforms:', 0, index_side_effect)]
                    try:
                        side_effect_pypath = i[i.rfind('mindspore'):]
                        replace_str = "    Side Effects:\n        ``Memory``\n\n"
                        if 'io' in key:
                            replace_str = replace_str.replace("Memory", "IO")
                        base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
                        with open(os.path.join(base_path, os.path.normpath(side_effect_pypath)), "r+", encoding="utf8") as q:
                            content_repl = q.read()
                            str_note_repl = str_notes + replace_str
                            if str_note_repl not in content_repl:
                                content_repl = content_repl.replace(str_notes, str_note_repl)
                                q.seek(0)
                                q.truncate()
                                q.write(content_repl)
                    except Exception as e:
                        pass

from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

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
copyright = '2022, MindSpore'
author = 'MindSpore'
# language = 'cn'
# The full version, including alpha/beta/rc tags
release = 'master'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
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
exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------

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
import anchor_mod
import nbsphinx_mod

sys.path.append(os.path.abspath('../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective

# -- Options for Texinfo output -------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../resource/numpy_objects.inv'),
}

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary, MsCnAutoSummary, MsCnPlatformAutoSummary, MsCnNoteAutoSummary


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

src_dir = os.path.join(os.getenv("MS_PATH"), 'docs/api/api_python')
des_sir = "./api_python"

if not exists(src_dir):
    logger.warning(f"不存在目录：{src_dir}！")
if os.path.exists(des_sir):
    shutil.rmtree(des_sir)
shutil.copytree(src_dir, des_sir)

target_dir="./api_python/ops/"
try:
    for filename in os.listdir(target_dir):
        newname = filename.replace("func_",'')
        os.rename(os.path.join(target_dir, filename),os.path.join(target_dir, newname))
except Exception as e:
    print(e)

rst_files = set([i.replace('.rst', '') for i in glob.glob('api_python/**/*.rst', recursive=True)])

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)
    app.add_directive('mscnautosummary', MsCnAutoSummary)
    app.add_directive('mscnplatformautosummary', MsCnPlatformAutoSummary)
    app.add_directive('mscnnoteautosummary', MsCnNoteAutoSummary)
    app.add_config_value('rst_files', set(), False)
    app.add_directive('includecode', IncludeCodeDirective)
    app.add_stylesheet('css/bootstrap.min.css')
    app.add_stylesheet('css/training.css')
    app.add_javascript('js/training.js')


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

src_release = os.path.join(os.getenv("MS_PATH"), 'RELEASE_CN.md')
des_release = "./RELEASE.md"
with open(src_release, "r", encoding="utf-8") as f:
    data = f.read()
content = re.findall("## [\s\S\n]*", data)[0]
content2 = re.findall("(## MindSpore Lite[\s\S\n]*?\n)## ", data)[0]
con = content.replace(content2, "")
result = con.replace('# MindSpore', '#', 1)
with open(des_release, "w", encoding="utf-8") as p:
    p.write("# Release Notes"+"\n\n")
    p.write(result)
