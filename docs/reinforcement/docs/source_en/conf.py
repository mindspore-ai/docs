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
import sys
import IPython
import re
import nbsphinx as nbs

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = '2023, MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
myst_enable_extensions = ["dollarmath", "amsmath"]

myst_update_mathjax = False

myst_heading_anchors = 4

extensions = [
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
mathjax_path = 'https://cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

mathjax_options = {
    'async':'async'
}

exclude_patterns = []

highlight_language = 'none'

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_search_language = 'en'

# Remove extra outputs for nbsphinx extension.
nbsphinx_source_re = re.compile(r"(app\.connect\('html-collect-pages', html_collect_pages\))")
nbsphinx_math_re = re.compile(r"(\S.*$)")
mod_path = os.path.abspath(nbs.__file__)
with open(mod_path, "r+", encoding="utf8") as f:
    contents = f.readlines()
    for num, line in enumerate(contents):
        _content_re = nbsphinx_source_re.search(line)
        if _content_re and "#" not in line:
            contents[num] = nbsphinx_source_re.sub(r"# \g<1>", line)
        if "mathjax_config = app.config" in line and "#" not in line:
            contents[num:num+10] = [nbsphinx_math_re.sub(r"# \g<1>", i) for i in contents[num:num+10]]
            break
    exec("".join(contents), nbs.__dict__)


sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code
