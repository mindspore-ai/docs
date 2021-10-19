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
        EXCLUDE_SYMBOLS = operator*
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

# Copy sourcefiles from mindspore repository to "../include/".
import json
import re
from sphinx.util import logging
logger = logging.getLogger(__name__)

ms_path = os.getenv("MS_PATH")
if os.path.exists("../include"):
    shutil.rmtree("../include")
os.mkdir("../include")

#Copy header files mapping .cmake files.
cmake_path = [("mindspore/lite/cmake/file_list.cmake", "${CORE_DIR}", "mindspore/core")]
header_files = []
for i in cmake_path:
    with open(os.path.join(ms_path, i[0])) as f:
        for j in f.readlines():
            re_str = i[1].replace("$", "\$").replace('{', '\{').replace('}', '\}') + r'.*?\.h'
            pattern_ = re.findall(re_str, j)
            if not pattern_:
                continue
            header_files.append(os.path.join(ms_path, pattern_[0].replace(i[1], i[-1])))

for file_ in header_files:
        target_dir = os.path.join("../include", os.path.normpath(re.sub(rf"^{ms_path}/(mindspore/)?", "", os.path.dirname(file_))))
        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            shutil.copy(file_, target_dir)
        except FileNotFoundError:
            logger.warning("头文件{} 没有找到,!".format(file_))

# Copy header files with specified path.
with open("./SourceFileNames.json") as f:
    hfile_dic = json.load(f)
    exclude_hfiles = hfile_dic.get("with-exclude")
    no_exclude_hfiles = hfile_dic.get("with-no-exclude")

    # Deal with with-no-exclude.
    hfile_no_exclude = []
    for i in no_exclude_hfiles.items():
        dir_name, hfile_list_ = i
        source_dir = os.path.join(ms_path, os.path.normpath(dir_name))
        for j in hfile_list_:
            if "*" in j:
                hfile_no_exclude.extend(glob.glob(os.path.join(source_dir, j), recursive=True))
            else:
                hfile_no_exclude.append(os.path.join(source_dir, j))

    # Deal with with-exclude.
    hfile_lists = []
    exclude_lists = []
    for i in exclude_hfiles.items():
        dir_name, pattern_ = i
        source_dir = os.path.join(ms_path, os.path.normpath(dir_name))
        for j in pattern_.get("pattern"):
            if "*" in j:
                hfile_lists.extend(glob.glob(os.path.join(source_dir, j), recursive=True))
            else:
                hfile_lists.append(os.path.join(source_dir, j))
        for exclude_ in pattern_.get("exclude"):
            if "*" in exclude_:
                exclude_lists.extend(glob.glob(os.path.join(source_dir, exclude_), recursive=True))
            else:
                exclude_lists.append(os.path.join(source_dir, exclude_))

    # Copy header files.
    hfile_with_exclude = list(set(hfile_lists).difference(set(exclude_lists)))
    all_hfiles = hfile_no_exclude + hfile_with_exclude
    for file_ in all_hfiles:
        target_dir = os.path.join("../include", os.path.normpath(re.sub(rf"^{ms_path}/(mindspore/)?", "", os.path.dirname(file_))))
        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            shutil.copy(file_, target_dir)
        except FileNotFoundError:
            logger.warning("头文件{} 没有找到,!".format(file_))

# Remove "MS_API" in classes.
files_copyed = glob.glob("../include/**/*.h")
for file in files_copyed:
    with open(file, "r+", encoding="utf8") as f:
        content = f.read()
        if "class MS_API " in content:
            content_new = content.replace("class MS_API", "class")
            f.seek(0)
            f.truncate()
            f.write(content_new)


sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code
