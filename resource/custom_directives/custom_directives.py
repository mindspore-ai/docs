"""Customized directives for sphinx."""
import re
import os
from docutils.parsers.rst import Directive, directives
from docutils import nodes
from sphinx.util import logging


logger = logging.getLogger(__name__)

class IncludeCodeDirective(Directive):
    """
    Include source file without docstring at the top of file.

    Example usage:

    In `.rst` files, it should be used as below:

    .. includecode:: path/to/sample_code.py
        :position: begin_label, end_label

    or

    .. includecode:: path/to/sample_code.py

    In `.md` files, it should be used as below:

    ```{includecode} path/to/sample_code.py
    ---
    position: begin_label, end_label
    ```

    or

    ```{includecode} path/to/sample_code.py
    ```
    """

    # defines the parameter the directive expects
    # directives.unchanged means you get the raw value from RST
    option_spec = {
                "position": directives.unchanged
                }
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    add_index = False

    docstring_pattern = r'^#?[\s\S]*?("""|\'\'\')[\s\S]*?("""|\'\'\')\n'
    docstring_regex = re.compile(docstring_pattern)

    def run(self):
        """`run` method for directive."""
        document = self.state.document
        env = document.settings.env
        filename = os.path.normpath((self.arguments[0]))
        sample_code_path = os.path.join(os.path.dirname(__file__), "../../docs/sample_code")
        file_path = os.path.join(sample_code_path, filename)

        try:
            text = open(file_path).read()
            matched_code = ""
            text_no_docstring = self.docstring_regex.sub('', text, count=1)
            if "position" not in self.options:
                matched_code = text_no_docstring
            else:
                start_label, stop_label = self.options["position"].split(", ")
                matched_regex = re.compile(r'# {}\n([\s\S]*?)# {}'.format(start_label, stop_label))
                matched_code = matched_regex.findall(text_no_docstring)[0]
            code_block = nodes.literal_block(text=matched_code)
            if not matched_code:
                logger.warning('{}: warning: {} could not get '\
                               'specified code string.'.format(env.docname, filename))
                raise Exception
            return [code_block]

        except FileNotFoundError:
            logger.warning('{}: WARNING: {} file not found.'.format(env.docname, filename))
        finally:
            logger.warning('{}: WARNING: {} parse error.'.format(env.docname, filename))
