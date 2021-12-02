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
import sys
sys.path.append(os.path.abspath('../_ext'))
from sphinx.ext import autodoc as sphinx_autodoc
from sphinx.util import inspect as sphinx_inspect
from sphinx.domains import python as sphinx_domain_python
from textwrap import dedent

import mindinsight
import mindconverter

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = '2020, MindSpore'
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
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
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

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# -- Options for Texinfo output -------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../numpy_objects.inv'),
}

# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
inspect_source_path = os.path.abspath(sphinx_inspect.__file__)
autodoc_source_re = re.compile(r"(\s+)args = self\.format_args\(\*\*kwargs\)")
inspect_source_code_str = """signature = inspect.signature(subject)"""
inspect_target_code_str = """signature = my_signature.signature(subject)"""
autodoc_source_code_str = """args = self.format_args(**kwargs)"""
is_autodoc_code_str = """args = args.replace("'", "")"""
anotation_repair_str = """\
        for i, param in enumerate(parameters):
            if isinstance(param.annotation, str) and param.name in annotations:
                parameters[i] = param.replace(annotation=annotations[param.name])
"""

with open(autodoc_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    if is_autodoc_code_str not in code_str:
        code_str_lines = code_str.split("\n")
        autodoc_target_code_str = None
        for line in code_str_lines:
            re_matched_str = autodoc_source_re.search(line)
            if re_matched_str:
                space_num = re_matched_str.group(1)
                autodoc_target_code_str = dedent("""\
                    {0}
                    {1}if type(args) != type(None):
                    {1}    {2}""".format(autodoc_source_code_str, space_num, is_autodoc_code_str))
                break
        if autodoc_target_code_str:
            code_str = code_str.replace(autodoc_source_code_str, autodoc_target_code_str)
            exec(code_str, sphinx_autodoc.__dict__)
with open(inspect_source_path, "r+", encoding="utf8") as g:
    code_str = g.read()
    code_str = code_str.replace(inspect_source_code_str, inspect_target_code_str)
    code_str = code_str.replace(anotation_repair_str, "")
    if "import my_signature" not in code_str:
        code_str = code_str.replace("import sys", "import sys\nimport my_signature")
    exec(code_str, sphinx_inspect.__dict__)

# remove extra space for default params for autodoc.
sphinx_domain_python_source_path = os.path.abspath(sphinx_domain_python.__file__)
python_code_source = """for argument in arglist.split(','):"""
python_code_target = """for argument in [" " + i if num > 1 else i for num,i in enumerate(arglist.split(", "))]:"""
with open(sphinx_domain_python_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    if python_code_target not in code_str:
        code_str = code_str.replace(python_code_source, python_code_target)
        exec(code_str, sphinx_domain_python.__dict__)


sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code
