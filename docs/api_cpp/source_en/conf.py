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
    'recommonmark',
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

html_static_path = ['_static']

html_search_language = 'en'

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

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
            ":protected-members:",
            ":private-members:"
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
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": "..",
    ############################################################################
    # Suggested optional arguments.                                            #
    ############################################################################
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": False,
    "verboseBuild": True,
    "exhaleDoxygenStdin": textwrap.dedent("""
        INPUT = ../include
        EXTRACT_ALL = NO
        HIDE_UNDOC_MEMBERS = YES
        HIDE_UNDOC_CLASSES = YES
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

pattern1 = """for l in sorted(lst):\n"""
exclude_namespaces = """\
                if l.name in "mindspore std":
                    continue\n"""

pattern2 = """            for title, kind in dump_order:\n"""
include_summaries = """\
            dump_order = [
                ("Namespaces", "namespace")]\n"""

pattern3 = "for page, is_orphan in [(unabridged_api, False), (orphan_api, True)]:\n"
insert_str = "                break\n"

pattern4 = "            for l in lst:\n"
insert2_str = """\
                if spec.spec_apis and l.kind == "class" and l.name not in spec.class_list:
                    continue\n"""

pattern5 = r"""openFile.write(textwrap.dedent('''
                {heading}
                {heading_mark}

            '''.format(
                heading=subsectionTitle,
                heading_mark=utils.heading_mark(
                    subsectionTitle,
                    configs.SUB_SUB_SECTION_HEADING_CHAR
                )
            )))"""
insert3_str = r"""if subsectionTitle != "Namespaces":
                openFile.write(textwrap.dedent('''
                    {heading}
                    {heading_mark}
    
                '''.format(
                    heading=subsectionTitle,
                    heading_mark=utils.heading_mark(
                        subsectionTitle,
                        configs.SUB_SUB_SECTION_HEADING_CHAR
                    )
                )))"""

pattern6 = """                    heading=nspace.title,\n"""
replace_str = """                    heading=nspace.title.replace("Namespace ", ""),\n"""

pattern7 = "needs_parameters = len(functions) > 1"
replace_str2 = "needs_parameters = False"

# Replacing link of function to specified name.
pattern8 = 'child_refid = member.attrs["refid"]\n'
replace_str3 = """\
# Additional
                            if child_kind != "class":
                                name = curr_name + "::" + child_name
                                function_counter[name] = function_counter.setdefault(name, 0) + 1
                                if function_counter[name] == 1:
                                    child_refid = (name + "-1").replace(":", "_1")
                                else:
                                    child_refid = (name + f"-{function_counter.get(name)}").replace(":", "_1")
                                function_counter[child_refid] = member.attrs["refid"]
                            else:
                                child_refid = member.attrs["refid"]
"""
pattern9 = "if candidate.refid == func_refid:"
replace_str4 = "if function_counter[candidate.refid] == func_refid:"
pattern10 = "class ExhaleRoot"
add_str = "function_counter = dict()\n"

# Fix error for add source file when parsing function.
pattern11 = '                            refid = memberdef.attrs["id"]\n'
replace_str5 = """\
                            refid_0 = memberdef.attrs["id"]
                            refid = refid_0
                            if refid_0 in list(function_counter.values()):
                                for key, value in function_counter.items():
                                    if refid_0 == value:
                                        refid = key
                                        break
"""

pattern12 = """node.link_name = "exhale_{kind}_{id}".format(kind=node.kind, id=unique_id)
            if unique_id.startswith(node.kind):
                node.file_name = "{id}.rst".format(id=unique_id)
            else:
                node.file_name = "{kind}_{id}.rst".format(kind=node.kind, id=unique_id)
"""
replace_str6 = """node.link_name = "exhale_{kind}_{id}".format(kind=node.kind, id=unique_id).replace("_1", "_").replace("__", "_")
            if unique_id.startswith(node.kind):
                node.file_name = "{id}.rst".format(id=unique_id).replace("_1", "_").replace("__", "_")
            else:
                node.file_name = "{kind}_{id}.rst".format(kind=node.kind, id=unique_id).replace("_1", "_").replace("__", "_")
"""

# Fix error function parameters for mindspore:dataset:Affine.
pattern13 = """\
for param in memberdef.find_all("param", recursive=False):
                    parameters.append(param.type.text)
"""
replace_str7 = """\
for param in memberdef.find_all("param", recursive=False):
                    if "M\\n[" in param.text and "]" in param.text:
                        param_name = param.text.replace("\\n", "").replace("M[", "[")
                    else:
                        param_name = param.type.text
                    parameters.append(param_name)
"""

with open(exh_file, "r+") as f:
    content = f.read()
    content_sub = content
    if "import spec\n" not in content:
        content_sub = content_sub.replace("import textwrap\n", "import textwrap\nimport spec\n")
    if exclude_namespaces not in content:
        content_sub = content_sub.replace(pattern1, pattern1 + exclude_namespaces)
    if include_summaries not in content:
        content_sub = content_sub.replace(pattern2, include_summaries + pattern2)
    if pattern3 + insert_str not in content:
        content_sub = content_sub.replace(pattern3, pattern3 + insert_str)
    if insert2_str not in content:
        content_sub = content_sub.replace(pattern4, pattern4 + insert2_str)
    if insert3_str not in content:
        content_sub = content_sub.replace(pattern5, insert3_str)
    if replace_str not in content:
        content_sub = content_sub.replace(pattern6, replace_str)
    if replace_str2 not in content:
        content_sub = content_sub.replace(pattern7, replace_str2)
    if replace_str3 not in content:
        content_sub = content_sub.replace(pattern8, replace_str3)
    if replace_str4 not in content:
        content_sub = content_sub.replace(pattern9, replace_str4)
    if add_str not in content:
        content_sub = content_sub.replace(pattern10, add_str + pattern10)
    if replace_str5 not in content:
        content_sub = content_sub.replace(pattern11, replace_str5)
    if replace_str6 not in content:
        content_sub = content_sub.replace(pattern12, replace_str6)
    if replace_str7 not in content:
        content_sub = content_sub.replace(pattern13, replace_str7)
    if content_sub != content:
        f.seek(0)
        f.truncate()
        f.write(content_sub)


# Copy sourcefiles from mindspore repository.
ms_path = os.getenv("MS_PATH")
if os.path.exists("../include"):
    shutil.rmtree("../include")
os.mkdir("../include")
with open("../_custom/SourceFileNames.txt") as f:
    contents = f.readlines()
    for i in contents:
        if i == "\n":
            continue
        shutil.copy(os.path.join(ms_path, i.strip().strip("\n")), "../include/")

