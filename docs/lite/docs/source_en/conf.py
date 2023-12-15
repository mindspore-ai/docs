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
import regex

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

exclude_patterns = []

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

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

if os.getenv("MS_PATH").split('/')[-1]:
    copy_repo = os.getenv("MS_PATH").split('/')[-1]
else:
    copy_repo = os.getenv("MS_PATH").split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

try:
    src_release = os.path.join(os.getenv("MS_PATH"), 'RELEASE.md')
    des_release = "./RELEASE.md"
    release_source = f'[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/{copy_repo}/blob/{branch}/' + 'RELEASE.md)\n'
    with open(src_release, "r", encoding="utf-8") as f:
        data = f.read()
    if len(re.findall("\n## (.*?)\n",data)) > 1:
        content = regex.findall("(\n## MindSpore Lite [\s\S\n]*?)\n## ", data, overlapped=True)
        repo_version = re.findall("\n## MindSpore Lite ([0-9]+?\.[0-9]+?)\.([0-9]+?)[ -]", content[0])[0]
        content_new = ''
        for i in content:
            if re.findall(f"\n## MindSpore Lite ({repo_version[0]}\.[0-9]+?)[ -]", i):
                content_new += i
        content = content_new
    else:
        content = re.findall("(\n## [\s\S\n]*)", data)
        content = content[0]
    #result = content[0].replace('# MindSpore', '#', 1)
    with open(des_release, "w", encoding="utf-8") as p:
        p.write("# Release Notes" + "\n\n" + release_source)
        p.write(content)
except Exception as e:
    print(e)
