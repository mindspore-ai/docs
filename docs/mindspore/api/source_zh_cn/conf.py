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
myst_enable_extensions = ["dollarmath", "amsmath"]

myst_update_mathjax = False

myst_heading_anchors = 4
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


# Repair error decorators defined in mindspore.
try:
    decorator_list = [("mindspore/common/_decorator.py", "deprecated",
                       "    def decorate(func):",
                       "    def decorate(func):\n\n        import functools\n\n        @functools.wraps(func)"),
                       ("mindspore/dataset/engine/graphdata.py","modify url",
                       "https://mindspore.cn/docs/programming_guide/zh-CN/master/load_dataset_gnn.html","https://mindspore.cn/docs/programming_guide/zh-CN/r1.6/load_dataset_gnn.html"),
                       ("mindspore/dataset/engine/datasets_user_defines.py","modify url",
                       "https://www.mindspore.cn/docs/programming_guide/en/master/dataset_loading.html#loading-user-defined-dataset","https://www.mindspore.cn/docs/programming_guide/en/r1.6/dataset_loading.html#loading-user-defined-dataset"),
                      ("mindspore/ops/operations/nn_ops.py", "modify math error.", ": math:`(", ":math:`("),
                      ("mindspore/common/initializer.py", "modify math error.", "text{fan_in}", "text{fan\_in}")
                     ]

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

image_specified = {"docs/api_img/*.png": "./api_python/ops/api_img",
                   "docs/api_img/dataset/*.png": "./api_python/dataset/api_img"}

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
