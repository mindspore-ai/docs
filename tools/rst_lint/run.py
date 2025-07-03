"""The restructuredtext linter."""
import os
import shutil
import sys
import restructuredtext_lint
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst.roles import register_generic_role, set_classes
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from sphinx import addnodes
from sphinx.ext.autodoc.directive import AutodocDirective
from sphinx.domains.changeset import VersionChange
from sphinx.domains.python import PyCurrentModule, PyModule
from sphinx.directives.other import TocTree
from sphinx.directives.code import LiteralInclude


class CustomDirective(Directive):
    """Base class of customized directives for python domains in sphinx."""
    has_content = True
    required_arguments = 1
    optional_arguments = 3
    final_argument_whitespace = True

    def run(self):
        """run method."""
        self.assert_has_content()
        text = '\n'.join(self.content)
        classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

class CustomDirectiveMethod(Directive):
    """Base class of customized directives for python method in sphinx."""
    has_content = True
    required_arguments = 1
    optional_arguments = 3
    final_argument_whitespace = True

    def run(self):
        """run method."""
        text = '\n'.join(self.content)
        classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

class CustomDirectiveFunc(Directive):
    """Base class of customized directives for python method in sphinx."""
    has_content = True
    required_arguments = 1
    optional_arguments = 3
    final_argument_whitespace = True

    def run(self):
        """run function."""
        text = '\n'.join(self.content)
        classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

class CustomDirectiveNoNested(CustomDirective):
    """Customizing CustomDirective with nonest."""
    def run(self):
        self.assert_has_content()
        text = '\n'.join(self.content)
        classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        return [node]


class Autoclass(AutodocDirective):
    """Customizing automodule."""
    def run(self):
        """run method."""
        text = '\n'.join(self.content)
        classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        return [node]


class Toctree(TocTree):
    """Customizing toctree."""

    def run(self):
        """run method."""
        text = '\n'.join(self.content)
        if self.arguments:
            classes = directives.class_option(self.arguments[0])
        else:
            classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

class CustomLiteralInclude(LiteralInclude):
    """Customizing toctree."""

    def run(self):
        """run method."""
        text = '\n'.join(self.content)
        if self.arguments:
            classes = directives.class_option(self.arguments[0])
        else:
            classes = []
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

class CurrentModule(PyCurrentModule):
    """Customizing currentmodule."""
    has_content = False
    required_arguments = 1
    optional_arguments = 3
    final_argument_whitespace = False

    def run(self):
        """run method."""
        return []

class CustomPyModule(PyModule):
    """Customizing currentmodule."""
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        """run method."""
        return []

class CustomVersionChange(VersionChange):
    """Customizing VersionChange."""
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        """run method."""
        node = addnodes.versionmodified()
        self.set_source_info(node)
        node['type'] = self.name
        node['version'] = self.arguments[0]
        if len(self.arguments) == 2:
            inodes, messages = self.state.inline_text(self.arguments[1],
                                                      self.lineno + 1)
            para = nodes.paragraph(self.arguments[1], '', *inodes, translatable=False)
            self.set_source_info(para)
            node.append(para)
        else:
            messages = []
        return [node]+messages

class CustomDirectiveSeeAlso(BaseAdmonition):
    """Customizing SeeAlso."""
    has_content = True
    final_argument_whitespace = True
    option_spec = {'class': directives.class_option,
                   'name': directives.unchanged}
    node_class = addnodes.seealso

    def run(self):
        set_classes(self.options)
        self.assert_has_content()
        text = '\n'.join(self.content)
        admonition_node = self.node_class(text, **self.options)
        self.add_name(admonition_node)
        if self.node_class is nodes.admonition:
            title_text = self.arguments[0]
            textnodes, messages = self.state.inline_text(title_text,
                                                         self.lineno)
            title = nodes.title(title_text, '', *textnodes)
            title.source, title.line = (
                self.state_machine.get_source_and_line(self.lineno))
            admonition_node += title
            admonition_node += messages
            if not 'classes' in self.options:
                admonition_node['classes'] += ['admonition-' +
                                               nodes.make_id(title_text)]
        self.state.nested_parse(self.content, self.content_offset,
                                admonition_node)
        return [admonition_node]

# Register directive.
register_directive('py:class', CustomDirective)
register_directive('py:method', CustomDirectiveMethod)
register_directive('py:function', CustomDirectiveFunc)
register_directive('py:property', CustomDirective)
register_directive('py:data', CustomDirective)
register_directive('py:obj', CustomDirective)
register_directive('py:module', CustomPyModule)
register_directive('automodule', Autoclass)
register_directive('autoclass', Autoclass)
register_directive('autofunction', Autoclass)
register_directive('toctree', Toctree)
register_directive('autosummary', CustomDirectiveNoNested)
register_directive('msplatformautosummary', CustomDirectiveNoNested)
register_directive('msplatwarnautosummary', CustomDirectiveNoNested)
register_directive('msnoteautosummary', CustomDirectiveNoNested)
register_directive('msmathautosummary', CustomDirectiveNoNested)
register_directive('mscnautosummary', CustomDirectiveNoNested)
register_directive('mscnplatformautosummary', CustomDirectiveNoNested)
register_directive('mscnnoteautosummary', CustomDirectiveNoNested)
register_directive('mscnmathautosummary', CustomDirectiveNoNested)
register_directive('mscnplatwarnautosummary', CustomDirectiveNoNested)
register_directive('mscnplataclnnautosummary', CustomDirectiveNoNested)
register_directive('currentmodule', CurrentModule)
register_directive('literalinclude', CustomLiteralInclude)
register_directive('deprecated', CustomVersionChange)
register_directive('seealso', CustomDirectiveSeeAlso)

# Register roles.
register_generic_role('class', nodes.literal)
register_generic_role('func', nodes.literal)
register_generic_role('attr', nodes.literal)
register_generic_role('doc', nodes.literal)
register_generic_role('py:obj', nodes.literal)

if __name__ == "__main__":
    try:
        decorator_list = [("restructuredtext_lint/cli.py", "cli.py")]

        base_path = os.path.dirname(os.path.dirname(restructuredtext_lint.__file__))
        for i in decorator_list:
            if os.path.exists(os.path.join(base_path, os.path.normpath(i[0]))):
                os.remove(os.path.join(base_path, os.path.normpath(i[0])))
                shutil.copy(os.path.join(os.path.dirname(__file__), i[1]),
                            os.path.join(base_path, os.path.normpath(i[0])))
    # pylint: disable=W0703
    except Exception:
        print('替换restructuredtext_lint安装包内容失败')
    # pylint: disable=C0412
    from restructuredtext_lint.cli import main

    sys.exit(main())
