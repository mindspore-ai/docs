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

# -- Project information -----------------------------------------------------

project = 'MindStudio'
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
    'python': ('https://docs.python.org/', '../../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../../resource/numpy_objects.inv'),
}


def setup(app):
    app.add_config_value('rst_files', set(), False)

# # add view
# import json

# if os.path.exists('../../../../tools/generate_html/version.json'):
#     with open('../../../../tools/generate_html/version.json', 'r+', encoding='utf-8') as f:
#         version_inf = json.load(f)
# elif os.path.exists('../../../../tools/generate_html/daily_dev.json'):
#     with open('../../../../tools/generate_html/daily_dev.json', 'r+', encoding='utf-8') as f:
#         version_inf = json.load(f)
# elif os.path.exists('../../../../tools/generate_html/daily.json'):
#     with open('../../../../tools/generate_html/daily.json', 'r+', encoding='utf-8') as f:
#         version_inf = json.load(f)

# if os.getenv("MSD_PATH").split('/')[-1]:
#     copy_repo = os.getenv("MSD_PATH").split('/')[-1]
# else:
#     copy_repo = os.getenv("MSD_PATH").split('/')[-2]

# branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo.replace('-', '_')][0]
# docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

# re_view = f"\n.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/" + \
#           f"resource/_static/logo_source.svg\n    :target: https://gitee.com/mindspore/{copy_repo}/blob/{branch}/"

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
# import anchor_mod
import nbsphinx_mod

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

# src_release = os.path.join(os.getenv("MFM_PATH"), 'RELEASE_CN.md')
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