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
sys.path.append(os.path.abspath('../_ext'))
import sphinx.ext.autosummary.generate as g
from sphinx.ext import autodoc as sphinx_autodoc

import mindformers

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = '2022, MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
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

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

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

html_search_options = {'dict': '../../resource/jieba.txt'}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../../resource/numpy_objects.inv'),
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

src_dir_mfl = os.path.join(os.getenv("MFM_PATH"), 'docs/api_python')

for i in os.listdir(src_dir_mfl):
    if os.path.isfile(os.path.join(src_dir_mfl,i)):
        if os.path.exists('./'+i):
            os.remove('./'+i)
        shutil.copy(os.path.join(src_dir_mfl,i),'./'+i)
    else:
        if os.path.exists('./'+i):
            shutil.rmtree('./'+i)
        shutil.copytree(os.path.join(src_dir_mfl,i),'./'+i)

sys.path.append(os.path.abspath('../../../../resource/custom_directives'))

from custom_directives import IncludeCodeDirective
from myautosummary import MsCnPlatformAutoSummary, MsPlatformAutoSummary

rst_files = set([i.replace('.rst', '') for i in glob.glob('./**/*.rst', recursive=True)])

def setup(app):
    app.add_directive('includecode', IncludeCodeDirective)
    app.add_directive('mscnplatformautosummary', MsCnPlatformAutoSummary)
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_config_value('rst_files', set(), False)

try:
    src_release = os.path.join(os.getenv("MT_PATH"), 'RELEASE_CN.md')
    des_release = "./RELEASE.md"
    with open(src_release, "r", encoding="utf-8") as f:
        data = f.read()
    content = re.findall("## [\s\S\n]*", data)
    result = content[0].replace('# MindSpore', '#', 1)
    with open(des_release, "w", encoding="utf-8") as p:
        p.write("# Release Notes"+"\n\n")
        p.write(result)
except Exception as e:
    print('release文件拷贝失败，原因是：',e)