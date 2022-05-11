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
import textwrap
import shutil
import glob

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
    'sphinx_markdown_tables',
    'myst_parser',
    'breathe',
    'exhale',
    'sphinx.ext.mathjax',
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

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_search_language = 'en'

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
import re

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
shutil.rmtree("../include/runtime/include/c_api")
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

# Remove "MS_API" in classes.
files_copyed = glob.glob("../include/**/*.h", recursive=True)
for file in files_copyed:
    with open(file, "r+", encoding="utf8") as f:
        content = f.read()
        if "MS_API" in content:
            content_new = content.replace("MS_API", "")
            f.seek(0)
            f.truncate()
            f.write(content_new)


sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code
