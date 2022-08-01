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
import textwrap
import shutil
import glob
from sphinx.ext import autodoc as sphinx_autodoc
import sphinx.ext.autosummary.generate as g

sys.path.append(os.path.abspath('../_ext'))
sys.path.append(os.path.abspath("../_custom"))
from exhale import graph as exh_graph

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
    'breathe',
    'exhale',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
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

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_search_language = 'en'

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

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary

sys.path.append(os.path.abspath('../../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)
    app.add_directive('includecode', IncludeCodeDirective)

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
        if "@classmethod" in source_code:
            all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
        else:
            all_params = re.sub("(self)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
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

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
import anchor_mod

# Tell sphinx what the primary language being documented is.
# primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
# highlight_language = 'cpp'

# Setup the breathe extension
breathe_projects = {
    "My Project": "./doxyoutput/xml"
}

breathe_default_project = "My Project"


def specificationsForKind(kind):
    '''
    For a given input ``kind``, return the list of reStructuredText specifications
    for the associated Breathe directive.
    '''
    # Change the defaults for .. doxygenclass:: and .. doxygenstruct::
    if kind == "class":
        return [
            ":members:",
            ":no-link:",
            # ":protected-members:",
            # ":private-members:"
        ]
    else:
        return []


# Use exhale's utility function to transform `specificationsForKind`
# defined above into something Exhale can use

from exhale import utils

exhale_args = {
    ############################################################################
    # These arguments are required.                                            #
    ############################################################################
    "containmentFolder": "./generate",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": "..",
    ############################################################################
    # Suggested optional arguments.                                            #
    ############################################################################
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": False,
    "verboseBuild": False,
    "exhaleDoxygenStdin": textwrap.dedent("""
        INPUT = ../include
        EXTRACT_ALL = NO
        HIDE_UNDOC_CLASSES = YES
        HIDE_UNDOC_MEMBERS = YES
        EXCLUDE_SYMBOLS = operator* GVAR*
    """),
    'contentsDirectives': False,

    ############################################################################
    # HTML Theme specific configurations.                                      #
    ############################################################################
    # Fix broken Sphinx RTD Theme 'Edit on GitHub' links
    # Search for 'Edit on GitHub' on the FAQ:
    #     http://exhale.readthedocs.io/en/latest/faq.html
    "pageLevelConfigMeta": ":gitee_url: https://gitee.com/mindspore/docs",
    ############################################################################
    # Individual page layout example configuration.                            #
    ############################################################################
    # Example of adding contents directives on custom kinds with custom title
    "contentsTitle": "Page Contents",
    "kindsWithContentsDirectives": ["class", "file", "namespace", "struct"],
    # Exclude PIMPL files from class hierarchy tree and namespace pages.
    # "listingExclude": [r".*Impl$"],
    ############################################################################
    # Main library page layout example configuration.                          #
    ############################################################################
    "afterTitleDescription": textwrap.dedent(u'''
        Welcome to the developer reference for the MindSpore C++ API.
    '''),
    # ... required arguments / other configs ...
    "customSpecificationsMapping": utils.makeCustomSpecificationsMapping(
        specificationsForKind
    )
}

# modify source code of exhale.
exh_file = os.path.abspath(exh_graph.__file__)

with open("../_custom/graph", "r", encoding="utf8") as f:
    source_code = f.read()

exec(source_code, exh_graph.__dict__)

# fix error of extra space for C++ API.
from sphinx.writers import html5 as sphinx_writer_html5

with open("../_custom/sphinx_writer_html5", "r", encoding="utf8") as f:
    source_code = f.read()

exec(source_code, sphinx_writer_html5.__dict__)

# fix position of "Return" for C++ API.
from sphinx.builders import html as sphinx_builder_html

with open("../_custom/sphinx_builder_html", "r", encoding="utf8") as f:
    source_code = f.read()

exec(source_code, sphinx_builder_html.__dict__)

import tarfile

# Copy source files of chinese python api from mindspore repository.
from sphinx.util import logging
import shutil
logger = logging.getLogger(__name__)
src_dir = os.path.join(os.getenv("MS_PATH"), 'docs/api/lite_api_python_en')

for i in os.listdir(src_dir):
    if '.' in i:
        if os.path.exists('./'+i):
            os.remove('./'+i)
        shutil.copy(os.path.join(src_dir,i),'./'+i)
    else:
        if os.path.exists('./'+i):
            shutil.rmtree('./'+i)
        shutil.copytree(os.path.join(src_dir,i),'./'+i)

lite_dir = './mindspore_lite'
if os.path.exists(lite_dir):
    shutil.rmtree(lite_dir)

import mindspore_lite

des_sir = "../include"
if os.path.exists(des_sir):
    shutil.rmtree(des_sir)

lite_package_path=os.getenv("LITE_PACKAGE_PATH", "null")
if lite_package_path == "null":
    print("LITE_PACKAGE_PATH: This environment variable does not exist")
    print("End of program")
    quit()
header_path = lite_package_path.split("/")[-1].split(".tar")[0]
save_path = "../"
os.makedirs(save_path, exist_ok=True)
t = tarfile.open(lite_package_path)
names = t.getnames()
for name in names:
    t.extract(name, save_path)
t.close()

source_path = "../" + header_path + "/"
source_runtime_include = os.path.join(source_path, "runtime/include")
target_runtime_include = "../include/runtime/include"
shutil.copytree(source_runtime_include, target_runtime_include)

source_converter_include = os.path.join(source_path, "tools/converter/include")
target_converter_include = "../include/converter/include"
shutil.copytree(source_converter_include, target_converter_include)

shutil.rmtree("../include/runtime/include/schema")
shutil.rmtree("../include/runtime/include/third_party")
shutil.rmtree("../include/converter/include/schema")
shutil.rmtree("../include/converter/include/third_party")
os.remove("../include/converter/include/api/types.h")

process = os.popen('pip show mindspore|grep Location')
output = process.read()
process.close()
mindspout = output.split(": ")[-1].strip()
source_dataset_dir = mindspout + "/mindspore/include/dataset/"
for file_ in os.listdir(source_dataset_dir):
    target_dataset_dir = "../include/runtime/include/dataset/"
    shutil.copy(source_dataset_dir+file_, target_dataset_dir)

for file_ in os.listdir("./api_cpp"):
    if file_.startswith("mindspore_") and file_ != 'mindspore_dataset.rst':
        os.remove("./api_cpp/"+file_)

fileList = []
for root, dirs, files in os.walk('../include/'):
    for fileObj in files:
        fileList.append(os.path.join(root, fileObj))

for file_name in fileList:
    file_data = ''
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('enum class', 'enum')
            file_data += line
    with open(file_name, 'w', encoding='utf-8') as p:
        p.write(file_data)

for file_name in fileList:
    file_data = ''
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub('^enum', 'enum class', line)
            file_data += line
    with open(file_name, 'w', encoding='utf-8') as p:
        p.write(file_data)

fileList1 = []
for root1, dirs1, files1 in os.walk('../include/converter/'):
    for fileObj1 in files1:
        fileList1.append(os.path.join(root1, fileObj1))

for file_name1 in fileList1:
    file_data1 = ''
    with open(file_name1, 'r', encoding='utf-8') as f:
        for line1 in f:
            line1 = re.sub(r'enum class (.*) {', r'enum class \1(converter) {', line1)
            file_data1 += line1
    with open(file_name1, 'w', encoding='utf-8') as p:
        p.write(file_data1)

fileList2 = []
for root2, dirs2, files2 in os.walk('../include/runtime/'):
    for fileObj2 in files2:
        fileList2.append(os.path.join(root2, fileObj2))

for file_name2 in fileList2:
    file_data2 = ''
    with open(file_name2, 'r', encoding='utf-8') as f:
        for line2 in f:
            line2 = re.sub(r'enum class (.*) {', r'enum class \1(runtime) {', line2)
            file_data2 += line2
    with open(file_name2, 'w', encoding='utf-8') as p:
        p.write(file_data2)

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code
