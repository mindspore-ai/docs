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
import os
import re
import shutil
import sys
import sphinx
from sphinx.ext import autodoc as sphinx_autodoc
import sphinx.ext.autosummary.generate as g

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

# -- Project information -----------------------------------------------------

project = 'MindSpore Transformers'
copyright = 'MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'dev'


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

# -- Options for HTML output -------------------------------------------------

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
    'python': ('https://docs.python.org/', '../../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../../resource/numpy_objects.inv'),
}

# overwriteautosummary_generate add view source for api.
with open('../_ext/overwriteautosummary_generate.txt', 'r', encoding="utf8") as f:
    exec(f.read(), g.__dict__)

# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
autodoc_source_re = re.compile(r'stringify_signature\(.*?\)')
get_param_func_str = r"""\
import re
import inspect as inspect_

def get_param_func(func):
    try:
        source_code = inspect_.getsource(func)
        all_params = ''
        if hasattr(func, '__dataclass_fields__'):
            for k, v in getattr(func, '__dataclass_fields__').items():
                if hasattr(v, 'default'):
                    if isinstance(v.default, str):
                        all_params += f"{k} = '{v.default}', "
                    else:
                        all_params += f"{k} = {v.default}, "
                else:
                    all_params += f"{k}, "
            all_params = all_params.strip(', ')
        else:
            if func.__doc__:
                source_code = source_code.replace(func.__doc__, '')
            all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
            if "@classmethod" in source_code or "def __new__" in source_code:
                all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
                if ',' in all_params_str[0][0]:
                    all_params = re.sub("(self|cls)(, |,)", '', all_params_str[0][0].replace("\n", ""))
            else:
                all_params = re.sub("(self)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
                if ',' in all_params_str[0][0]:
                    all_params = re.sub("(self)(, |,)", '', all_params_str[0][0].replace("\n", ""))
        return all_params
    except:
        return ''

def get_obj(obj):
    if getattr(obj, '__dataclass_fields__', None):
        return obj

    if isinstance(obj, type):
        try:
            test_source = inspect_.getsource(obj.__init__)
        except:
            return obj.__new__
        obj_init = getattr(obj, '__init__', None)
        if obj.__name__ not in str(obj_init) and hasattr(obj, '__new__'):
            return obj.__new__
        return obj.__init__

    return obj
"""

with open(autodoc_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    code_str = autodoc_source_re.sub('"(" + get_param_func(get_obj(self.object)) + ")"', code_str, count=0)
    exec(get_param_func_str, sphinx_autodoc.__dict__)
    exec(code_str, sphinx_autodoc.__dict__)

# add @functools.wraps
try:
    decorator_list = [("mindformers/tools/logger.py", "__call__", "wrapper"),
                      ("mindformers/version_control.py", "get_lazy_inline", "decorator")]

    base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
    for i in decorator_list:
        with open(os.path.join(base_path, os.path.normpath(i[0])), "r+", encoding="utf8") as f:
            content = f.read()
            new_content = re.sub('(import .*\n)', r'\1import functools\n', content, 1)
            new_content = re.sub(f'def ({i[1]})\((.*?)\):\n(((?!wraps).|\n)*?)([ ]+?)def {i[2]}\(',
                             rf'def \1(\2):\n\3\5@functools.wraps(\2)\n\5def {i[2]}(', new_content)
            new_content = re.sub('@functools.wraps\((self|cls),[ ]*', r'@functools.wraps(', new_content)
            if new_content != content:
                f.seek(0)
                f.truncate()
                f.write(new_content)
except:
    pass

import mindformers

# Copy source files of chinese python api from golden-stick repository.
from sphinx.util import logging
import shutil
logger = logging.getLogger(__name__)

src_dir_api = os.path.join(os.getenv("MFM_PATH"), 'docs/api/api_python_en')
moment_dir=os.path.dirname(__file__)

for root,dirs,files in os.walk(src_dir_api):
    for file in files:
        if os.path.exists(os.path.join(moment_dir,file)):
            os.remove(os.path.join(moment_dir,file))
        shutil.copy(os.path.join(src_dir_api,file),os.path.join(moment_dir,file))

if os.path.exists('./mindformers.experimental.rst'):
    os.remove('./mindformers.experimental.rst')

# get params for add view source
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

if os.getenv("MFM_PATH").split('/')[-1]:
    copy_repo = os.getenv("MFM_PATH").split('/')[-1]
else:
    copy_repo = os.getenv("MFM_PATH").split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo.replace('-','_')][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]
cst_module_name = 'mindformers'
repo_whl = 'mindformers'
giturl = 'https://gitee.com/mindspore/'

def setup(app):
    app.add_config_value('docs_branch', '', True)
    app.add_config_value('branch', '', True)
    app.add_config_value('cst_module_name', '', True)
    app.add_config_value('copy_repo', '', True)
    app.add_config_value('giturl', '', True)
    app.add_config_value('repo_whl', '', True)

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
# import anchor_mod
import nbsphinx_mod

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

# src_release = os.path.join(os.getenv("MFM_PATH"), 'RELEASE.md')
# des_release = "./RELEASE.md"
# with open(src_release, "r", encoding="utf-8") as f:
#     data = f.read()
# if len(re.findall("\n## (.*?)\n",data)) > 1:
#     content = re.findall("(## [\s\S\n]*?)\n## ", data)
# else:
#     content = re.findall("(## [\s\S\n]*)", data)
# #result = content[0].replace('# MindSpore', '#', 1)
# with open(des_release, "w", encoding="utf-8") as p:
#     p.write("# Release Notes"+"\n\n")
#     p.write(content[0])