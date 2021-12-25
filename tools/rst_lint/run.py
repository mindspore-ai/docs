"""The restructuredtext linter."""
import sys
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives import register_directive
from docutils.parsers.rst.roles import register_generic_role
from sphinx.ext.autodoc.directive import AutodocDirective
from sphinx.domains.python import PyCurrentModule
from sphinx.directives.other import TocTree
from restructuredtext_lint.cli import main


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


class CurrentModule(PyCurrentModule):
    """Customizing currentmodule."""
    has_content = False
    required_arguments = 1
    optional_arguments = 3
    final_argument_whitespace = False

    def run(self):
        """run method."""
        return []


# Register directive.
register_directive('py:class', CustomDirective)
register_directive('py:method', CustomDirective)
register_directive('py:function', CustomDirective)
register_directive('py:property', CustomDirective)
register_directive('py:data', CustomDirective)
register_directive('py:obj', CustomDirective)
register_directive('automodule', Autoclass)
register_directive('autoclass', Autoclass)
register_directive('autofunction', Autoclass)
register_directive('toctree', Toctree)
register_directive('autosummary', CustomDirectiveNoNested)
register_directive('msplatformautosummary', CustomDirectiveNoNested)
register_directive('msnoteautosummary', CustomDirectiveNoNested)
register_directive('cnmsautosummary', CustomDirectiveNoNested)
register_directive('cnmsplatformautosummary', CustomDirectiveNoNested)
register_directive('cnmsnoteautosummary', CustomDirectiveNoNested)
register_directive('currentmodule', CurrentModule)

# Register roles.
register_generic_role('class', nodes.literal)
register_generic_role('func', nodes.literal)
register_generic_role('doc', nodes.literal)
register_generic_role('py:obj', nodes.literal)

if __name__ == "__main__":
    sys.exit(main())
