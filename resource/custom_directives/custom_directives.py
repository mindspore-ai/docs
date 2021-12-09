"""Customized directives for sphinx."""
import re
import os
from functools import reduce
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
        :position: begin_label, end_label|begin_labe2, end_labe2|...

    or

    .. includecode:: path/to/sample_code.py

    In `.md` files, it should be used as below:

    ```{eval-rst}
    .. includecode:: path/to/sample_code.py
        :position: begin_label, end_label|begin_labe2, end_labe2|...
    ```

    or

    ```{eval-rst}
    .. includecode:: path/to/sample_code.py
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
                matched_list = []
                positions = self.options["position"]
                positions_list = positions.replace(' ', '').split('|')
                positions_list = [i.split(',') for i in positions_list]
                exclude_ = reduce(lambda x, y: x + y, positions_list)
                exclude_ = ['# '+i+'\n' for i in exclude_]
                for position in positions_list:
                    start_label, stop_label = position
                    matched_regex = re.compile(r'# {}\n([\s\S]*?)# {}'.format(start_label, stop_label))
                    space_start_re = re.compile(r'^( +)')
                    matched_str = matched_regex.findall(text_no_docstring)[0]
                    space_start = space_start_re.findall(matched_str)
                    if space_start:
                        space_start = space_start[0]
                        matched_ = matched_str.split('\n')
                        matched_ = [i.split(space_start, 1)[-1] for i in matched_]
                        matched_str = '\n'.join(matched_)
                    matched_list.append(matched_str)
                matched_code = ''.join(matched_list)
                for i in exclude_:
                    matched_code = matched_code.replace(i, '')
                matched_code_list = matched_code.split('\n')
                result = []
                header_list = []
                if matched_code_list[0].startswith('from') or matched_code_list[0].startswith('import'):
                    for i in matched_code_list:
                        starts_ = re.findall(r'^(import|from|sys\.)', i)
                        if starts_:
                            header_list.append(i)
                        else:
                            result.append(i)
                    matched_code = '\n'.join(header_list) + '\n\n' + '\n'.join(result).lstrip('\n')

            code_block = nodes.literal_block(text=matched_code)
            if not matched_code:
                logger.warning('{}: warning:{}: there is no code to be found.'.format(env.docname, self.lineno))
                return []
            return [code_block]

        except FileNotFoundError:
            logger.warning('{}: WARNING: {}: {} file not found.'.format(env.docname, self.lineno, filename))
            return []
        except IndexError:
            logger.warning('{}: WARNING: {}: has error format.'.format(env.docname, self.lineno))
            return []
