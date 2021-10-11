"""The restructuredtext linter."""
import sys
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives import register_directive
from restructuredtext_lint.cli import main


class CustomDirective(Directive):
    """Base class of customized directives for python domains in sphinx."""
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    def run(self):
        """run method."""
        self.assert_has_content()
        text = '\n'.join(self.content)

        try:
            if self.arguments:
                classes = directives.class_option(self.arguments[0])
            else:
                classes = []
        except ValueError:
            raise self.error(
                'Invalid class attribute value for "%s" directive: "%s".'
                % (self.name, self.arguments[0]))
        node = nodes.container(text)
        node['classes'].extend(classes)
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class PyClass(CustomDirective):
    """Customizing py:class."""


class PyMethod(CustomDirective):
    """Customizing py:method."""


class PyFunction(CustomDirective):
    """Customizing py:function."""


class PyProperty(CustomDirective):
    """Customizing py:property."""


register_directive('py:class', PyClass)
register_directive('py:method', PyMethod)
register_directive('py:function', PyFunction)
register_directive('py:property', PyProperty)

if __name__ == "__main__":
    sys.exit(main())
