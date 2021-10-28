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
import sphinx
sys.path.append(os.path.abspath('../_ext'))
import sphinx.ext.autosummary.generate as g
from sphinx.ext import autodoc as sphinx_autodoc
from sphinx.util import inspect as sphinx_inspect
from sphinx.domains import python as sphinx_domain_python
from textwrap import dedent


# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = '2021, MindSpore'
author = 'MindSpore'

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

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)

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
inspect_source_path = os.path.abspath(sphinx_inspect.__file__)
autodoc_source_re = re.compile(r"(\s+)args = self\.format_args\(\*\*kwargs\)")
inspect_source_code_str = """signature = inspect.signature(subject)"""
inspect_target_code_str = """signature = my_signature.signature(subject)"""
autodoc_source_code_str = """args = self.format_args(**kwargs)"""
is_autodoc_code_str = """args = args.replace("'", "")"""
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
    if inspect_target_code_str not in code_str:
        code_str = code_str.replace(inspect_source_code_str, inspect_target_code_str)
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

# Repair error decorators defined in mindspore.
try:
    decorator_list = [("mindspore/common/_decorator.py", "deprecated",
                       "    def decorate(func):",
                       "    def decorate(func):\n\n        import functools\n\n        @functools.wraps(func)"),
                      ("mindspore/nn/optim/optimizer.py", "opt_init_args_register",
                       "    def deco(self, *args, **kwargs):",
                       "\n\n    import functools\n\n    @functools.wraps(fn)\n    def deco(self, *args, **kwargs):"),
                      ("mindspore/ops/primitive.py", "prim_attr_register",
                       "    def deco(self, *args, **kwargs)",
                       "\n\n    import functools\n\n    @functools.wraps(fn)\n    def deco(self, *args, **kwargs)"),
                       ("mindspore/nn/layer/basic.py", "Repair error comments for mindspore.nn.Pad.",
                       "paddings[0][1] = 1 + 3 + 1 = 4.", "paddings[0][1] = 1 + 3 + 1 = 5."),
                      ("mindspore/nn/layer/basic.py", "Repair error comments for mindspore.nn.Pad.",
                       "so output.shape is (4, 7)", "so output.shape is (5, 7)")]

    base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
    for i in decorator_list:
        with open(os.path.join(base_path, os.path.normpath(i[0])), "r+", encoding="utf8") as f:
            content = f.read()
            if i[3] not in content:
                content = content.replace(i[2], i[3])
                f.seek(0)
                f.truncate()
                f.write(content)
except:
    pass

import mindspore


sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

# Copy images from mindspore repository to sphinx workdir before running.
import glob
import shutil
from sphinx.util import logging
logger = logging.getLogger(__name__)

image_specified = {"docs/api_img/*.png": "./api_python/ops/api_img"}


for img in image_specified.keys():
    des_dir = os.path.normpath(image_specified[img])
    try:
        if "*" in img:
            imgs = glob.glob(os.path.join(os.getenv("MS_PATH"), os.path.normpath(img)))
            if not imgs:
                continue
            if not os.path.exists(des_dir):
                os.makedirs(des_dir)
            for i in imgs:
                shutil.copy(i, des_dir)
        else:
            img_fullpath = os.path.join(os.getenv("MS_PATH"), des_dir)
            if os.path.exists(img_fullpath):
                shutil.copy(img_fullpath, des_dir)
    except:
        logger.warning(f"{img} deal failed!")
