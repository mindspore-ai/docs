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
import shutil
import sys
import re
from sphinx.search import jssplitter as sphinx_split
from sphinx import errors as searchtools_path


# -- Project information -----------------------------------------------------

project = 'MindSpore Lite'
copyright = 'MindSpore'
author = 'MindSpore Lite'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
myst_enable_extensions = ["dollarmath", "amsmath"]


myst_heading_anchors = 5
extensions = [
    'myst_parser',
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

# -- Options for HTML output -------------------------------------------------

language = 'zh_CN'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#modify layout.html for sphinx_rtd_theme.
import sphinx_rtd_theme
layout_target = os.path.join(os.path.dirname(sphinx_rtd_theme.__file__), 'layout.html')
layout_src = '../../../../resource/_static/layout.html'
if os.path.exists(layout_target):
    os.remove(layout_target)
shutil.copy(layout_src, layout_target)

html_search_language = 'zh'

html_search_options = {'dict': '../../../resource/jieba.txt'}

html_static_path = ['_static']

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
# import anchor_mod

sys.path.append(os.path.abspath('../../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective

def setup(app):
    app.add_js_file('js/lite.js')
    app.add_directive('includecode', IncludeCodeDirective)

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

# try:
#     src_release = os.path.join(os.getenv("MS_PATH"), 'RELEASE_CN.md')
#     des_release = "./RELEASE.md"
#     with open(src_release, "r", encoding="utf-8") as f:
#         data = f.read()
#     content = re.findall("(## MindSpore Lite[\s\S\n]*?\n)## ", data)
#     with open(des_release, "w", encoding="utf-8") as p:
#         p.write("# Release Notes"+"\n\n")
#         p.write(content[0])
# except Exception as e:
#     print(e)