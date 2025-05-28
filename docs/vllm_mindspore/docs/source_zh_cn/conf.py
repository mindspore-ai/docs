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
import shutil
import sys
import IPython
import re
import sphinx
sys.path.append(os.path.abspath('../_ext'))
import sphinx.ext.autosummary.generate as g
from sphinx.ext import autodoc as sphinx_autodoc

# -- Project information -----------------------------------------------------

project = 'vLLM MindSpore'
copyright = 'MindSpore'
author = 'vLLM MindSpore'

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

smartquotes_action = 'De'

exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

autosummary_generate_overwrite = False

# -- Options for HTML output -------------------------------------------------

# Reconstruction of sphinx auto generated document translation.
language = 'zh_CN'
locale_dirs = ['../../../../resource/locale/']
gettext_compact = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_search_language = 'zh'

html_search_options = {'dict': '../../../resource/jieba.txt'}

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
# import anchor_mod
import nbsphinx_mod

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../../resource/numpy_objects.inv'),
}

from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

from sphinx.ext import viewcode
with open('../_ext/overwriteviewcode.txt', 'r', encoding="utf8") as f:
    exec(f.read(), viewcode.__dict__)

with open('../_ext/customdocumenter.txt', 'r', encoding="utf8") as f:
    exec(f.read(), sphinx_autodoc.__dict__)

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

def get_param_func(func):
    try:
        source_code = inspect_.getsource(func)
        if func.__doc__:
            source_code = source_code.replace(func.__doc__, '')
        all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
        all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", "").replace("'", "\""))
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

# Copy source files of chinese python api from mindscience repository.
from sphinx.util import logging
logger = logging.getLogger(__name__)

# copy_path = 'docs/api_python/mindchemistry'
# src_dir = os.path.join(os.getenv("VLLM_PATH"), copy_path)

copy_list = []

present_path = os.path.dirname(__file__)

# for i in os.listdir(src_dir):
#     if os.path.isfile(os.path.join(src_dir,i)):
#         if os.path.exists('./'+i):
#             os.remove('./'+i)
#         shutil.copy(os.path.join(src_dir,i),'./'+i)
#         copy_list.append(os.path.join(present_path,i))
#     else:
#         if os.path.exists('./'+i):
#             shutil.rmtree('./'+i)
#         shutil.copytree(os.path.join(src_dir,i),'./'+i)
#         copy_list.append(os.path.join(present_path,i))

# add view
import json

with open('../../../../tools/generate_html/daily.json', 'r+', encoding='utf-8') as f:
    version_inf = json.load(f)

# if os.getenv("VLLM_PATH").split('/')[-1]:
#     copy_repo = os.getenv("VLLM_PATH").split('/')[-1]
# else:
#     copy_repo = os.getenv("VLLM_PATH").split('/')[-2]

# import pdb
# pdb.set_trace()

# branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo][0]
# docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

# re_view = f"\n.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/" + \
#           f"resource/_static/logo_source.svg\n    :target: https://gitee.com/mindspore/{copy_repo}/blob/{branch}/"

# for cur, _, files in os.walk(present_path):
#     for i in files:
#         flag_copy = 0
#         if i.endswith('.rst'):
#             for j in copy_list:
#                 if j in cur:
#                     flag_copy = 1
#                     break
#             if os.path.join(cur, i) in copy_list or flag_copy:
#                 try:
#                     with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
#                         content = f.read()
#                         new_content = content
#                         if '.. include::' in content and '.. automodule::' in content:
#                             continue
#                         if 'autosummary::' not in content and "\n=====" in content:
#                             re_view_ = re_view + copy_path + cur.split(present_path)[-1] + '/' + i + \
#                                        '\n    :alt: 查看源文件\n\n'
#                             new_content = re.sub('([=]{5,})\n', r'\1\n' + re_view_, content, 1)
#                             print("re_view_")
#                             print(re_view_)
#                         if new_content != content:
#                             f.seek(0)
#                             f.truncate()
#                             f.write(new_content)
#                 except Exception:
#                     print(f'打开{i}文件失败')


# import vllm_mindspore

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

sys.path.append(os.path.abspath('../../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective
from myautosummary import MsPlatformAutoSummary, MsCnPlatformAutoSummary

rst_files = set([i.replace('.rst', '') for i in glob.glob('./**/*.rst', recursive=True)])

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('mscnplatformautosummary', MsCnPlatformAutoSummary)
    app.add_directive('includecode', IncludeCodeDirective)
    app.add_config_value('rst_files', set(), False)

src_release = "./release_notes/release_notes.md"
des_release = "./RELEASE.md"
with open(src_release, "r", encoding="utf-8") as f:
    data = f.read()
if len(re.findall("\n## (.*?)\n",data)) > 1:
    content = re.findall("(## [\s\S\n]*?)\n## ", data)
else:
    content = re.findall("(## [\s\S\n]*)", data)

with open(des_release, "w", encoding="utf-8") as p:
    p.write("# Release Notes"+"\n\n")
    p.write(content[0])

os.makedirs(os.path.join(present_path, "../build_zh_cn/html/"), exist_ok=True)
shutil.copy(os.path.join(present_path, "arch.cn.png"), os.path.join(present_path, "../build_zh_cn/html/"))
