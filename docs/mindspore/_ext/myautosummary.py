"""Customized autosummary directives for sphinx."""
import os
import re
import inspect
import importlib
from typing import List, Tuple
from docutils.nodes import Node
from sphinx.locale import __
from sphinx.ext.autosummary import Autosummary, posixpath, addnodes, logger, Matcher, autosummary_toc, get_import_prefixes_from_env
from sphinx.ext.autosummary import mock, StringList, ModuleType, get_documenter, ModuleAnalyzer, PycodeError, mangle_signature
from sphinx.ext.autosummary import import_by_name, extract_summary, autosummary_table, nodes, switch_source_input, rst
from sphinx.ext.autodoc.directive import DocumenterBridge, Options


class MsAutosummary(Autosummary):
    """
    Inherited from sphinx's autosummary, add titles and a column for the generated table.
    """

    def init(self):
        """
        init method
        """
        self.find_doc_name = ""
        self.third_title = ""
        self.default_doc = ""
        self.find_doc_name_fourth = ""
        self.fourth_title = ""
        self.default_doc_fourth = ""

    def extract_env_summary(self, doc: List[str]) -> str:
        """Extract env summary from docstring."""
        env_sum = self.default_doc
        for i, piece in enumerate(doc): # 行索引和行注释内容
            if piece.startswith(self.find_doc_name):
                env_sum = doc[i+1][4:] # 支持平台
        return env_sum

    def extract_ops_summary(self, doc: List[str]) -> str:
        """Extract env summary from docstring."""
        env_sum = ""
        end_doc = doc[-2]
        if 'Refer to' in end_doc and 'for more details' in end_doc:
            env_sum = end_doc
        return env_sum

    def extract_env_warn(self, doc: List[str]) -> str:
        """Extract env note from docstring."""
        flag_warn = 0
        env_warn = ''
        indent = 0
        for piece in doc:
            if piece == '.. rubric:: Examples':
                break
            if flag_warn and piece != '' and (len(piece) - len(piece.lstrip())) >= indent+3:
                if piece.lstrip().startswith('- '):
                    env_warn += piece.lstrip().lstrip('- ')+' '
                else:
                    env_warn += piece.lstrip()+' '
            elif piece == '':
                continue
            elif (len(piece) - len(piece.lstrip())) <= indent+3 and flag_warn:
                break
            if piece.startswith(self.find_doc_name_fourth):
                flag_warn = 1
                indent = len(piece) - len(piece.lstrip())
        if not env_warn:
            env_warn = "None"
        return env_warn.rstrip()

    def run(self):
        """
        run method
        """
        self.init()
        self.bridge = DocumenterBridge(self.env, self.state.document.reporter,
                                       Options(), self.lineno, self.state)

        names = [x.strip().split()[0] for x in self.content
                 if x.strip() and re.search(r'^[~a-zA-Z_]', x.strip()[0])]
        items = self.get_items(names)
        teble_nodes = self.get_table(items)

        if 'toctree' in self.options:
            dirname = posixpath.dirname(self.env.docname)

            tree_prefix = self.options['toctree'].strip()
            docnames = []
            excluded = Matcher(self.config.exclude_patterns)
            for item in items:
                docname = posixpath.join(tree_prefix, item[3])
                docname = posixpath.normpath(posixpath.join(dirname, docname))
                if docname not in self.env.found_docs:
                    location = self.state_machine.get_source_and_line(self.lineno)
                    if excluded(self.env.doc2path(docname, None)):
                        msg = __('autosummary references excluded document %r. Ignored.')
                    else:
                        msg = __('autosummary: stub file not found %r. '
                                 'Check your autosummary_generate setting.')
                    logger.warning(msg, item[3], location=location)
                    continue
                docnames.append(docname)

            if docnames:
                tocnode = addnodes.toctree()
                tocnode['includefiles'] = docnames
                tocnode['entries'] = [(None, docn) for docn in docnames]
                tocnode['maxdepth'] = -1
                tocnode['glob'] = None
                teble_nodes.append(autosummary_toc('', '', tocnode))
        return teble_nodes

    def get_items(self, names: List[str]) -> List[Tuple[str, str, str, str, str]]:
        """Try to import the given names, and return a list of
        ``[(name, signature, summary_string, real_name, env_summary), ...]``.
        """
        prefixes = get_import_prefixes_from_env(self.env)
        items = []  # type: List[Tuple[str, str, str, str, str]]
        max_item_chars = 50

        for name in names:
            display_name = name
            if name.startswith('~'):
                name = name[1:]
                display_name = name.split('.')[-1]
            try:
                with mock(self.config.autosummary_mock_imports):
                    real_name, obj, parent, modname = import_by_name(name, prefixes=prefixes)
            except ImportError:
                logger.warning(__('failed to import %s'), name)
                if self.fourth_title:
                    items.append((name, '', '', name, '', ''))
                else:
                    items.append((name, '', '', name, ''))
                continue

            self.bridge.result = StringList()  # initialize for each documenter
            full_name = real_name
            if not isinstance(obj, ModuleType):
                # give explicitly separated module name, so that members
                # of inner classes can be documented
                full_name = modname + '::' + full_name[len(modname) + 1:]
            # NB. using full_name here is important, since Documenters
            #     handle module prefixes slightly differently
            doccls = get_documenter(self.env.app, obj, parent)
            documenter = doccls(self.bridge, full_name)

            if not documenter.parse_name():
                logger.warning(__('failed to parse name %s'), real_name)
                if self.fourth_title:
                    items.append((display_name, '', '', real_name, '', ''))
                else:
                    items.append((display_name, '', '', real_name, ''))
                continue
            if not documenter.import_object():
                logger.warning(__('failed to import object %s'), real_name)
                if self.fourth_title:
                    items.append((display_name, '', '', real_name, '', ''))
                else:
                    items.append((display_name, '', '', real_name, ''))
                continue
            if documenter.options.members and not documenter.check_module():
                continue

            # try to also get a source code analyzer for attribute docs
            try:
                documenter.analyzer = ModuleAnalyzer.for_module(
                    documenter.get_real_modname())
                # parse right now, to get PycodeErrors on parsing (results will
                # be cached anyway)
                documenter.analyzer.find_attr_docs()
            except PycodeError as err:
                logger.debug('[autodoc] module analyzer failed: %s', err)
                # no source file -- e.g. for builtin and C modules
                documenter.analyzer = None

            # -- Grab the signature

            try:
                sig = documenter.format_signature(show_annotation=False)
            except TypeError:
                # the documenter does not support ``show_annotation`` option
                sig = documenter.format_signature()

            if not sig:
                sig = ''
            else:
                max_chars = max(10, max_item_chars - len(display_name))
                sig = mangle_signature(sig, max_chars=max_chars)

            # -- Grab the summary

            documenter.add_content(None)
            if '.ops.' in display_name:
                try:
                    api_obj = get_api(display_name)
                    if not api_obj:
                        logger.warning(f"not api obj: {name}")
                        display_name_path = ""
                    else:
                        display_name_path = inspect.getsourcefile(api_obj)
                # pylint: disable=W0702
                except:
                    display_name_path = ""
                if 'mindspore/ops/auto_generate/' in display_name_path:
                    env_sum = self.get_refer_platform(display_name)
                    summary = self.extract_ops_summary(self.bridge.result.data[:])
                    if not summary:
                        summary = extract_summary(self.bridge.result.data[:], self.state.document)
                    if not env_sum:
                        env_sum = self.extract_env_summary(self.bridge.result.data[:])
                else:
                    summary = extract_summary(self.bridge.result.data[:], self.state.document)
                    env_sum = self.extract_env_summary(self.bridge.result.data[:])
            else:
                summary = extract_summary(self.bridge.result.data[:], self.state.document)
                env_sum = self.extract_env_summary(self.bridge.result.data[:])
            if '.mint.' in display_name:
                env_sum = "``Ascend``"
            env_warn = self.extract_env_warn(self.bridge.result.data[:])
            if self.fourth_title:
                items.append((display_name, sig, summary, real_name, env_sum, env_warn))
            else:
                items.append((display_name, sig, summary, real_name, env_sum))

        return items

    def get_table(self, items: List[Tuple[str, str, str, str, str]]) -> List[Node]:
        """Generate a proper list of table nodes for autosummary:: directive.

        *items* is a list produced by :meth:`get_items`.
        """
        table_spec = addnodes.tabular_col_spec()
        table_spec['spec'] = r'\X{1}{2}\X{1}{2}'

        table = autosummary_table('')
        real_table = nodes.table('', classes=['longtable'])
        table.append(real_table)
        if not self.fourth_title:
            group = nodes.tgroup('', cols=3)
            real_table.append(group)
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=70))
            group.append(nodes.colspec('', colwidth=30))
        else:
            group = nodes.tgroup('', cols=4)
            real_table.append(group)
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=50))
            group.append(nodes.colspec('', colwidth=25))
            group.append(nodes.colspec('', colwidth=25))
        body = nodes.tbody('')
        group.append(body)

        def append_row(*column_texts: str) -> None:
            row = nodes.row('', color="red")
            source, line = self.state_machine.get_source_and_line()
            for text in column_texts:
                node = nodes.paragraph('')
                vl = StringList()
                vl.append(text, '%s:%d:<autosummary>' % (source, line))
                with switch_source_input(self.state, vl):
                    self.state.nested_parse(vl, 0, node)
                    try:
                        if isinstance(node[0], nodes.paragraph):
                            node = node[0]
                    except IndexError:
                        pass
                    row.append(nodes.entry('', node))
            body.append(row)

        # add table's title
        if not self.fourth_title:
            append_row("**API Name**", "**Description**", self.third_title)
            for name, sig, summary, real_name, env_sum in items:
                qualifier = 'obj'
                if 'nosignatures' not in self.options:
                    col1 = ':%s:`%s <%s>`\\ %s' % (qualifier, name, real_name, rst.escape(sig))
                else:
                    col1 = ':%s:`%s <%s>`' % (qualifier, name, real_name)
                col2 = summary
                col3 = env_sum
                append_row(col1, col2, col3)
        else:
            append_row("**API Name**", "**Description**", self.third_title, self.fourth_title)
            for name, sig, summary, real_name, env_sum, env_warn in items:
                qualifier = 'obj'
                if 'nosignatures' not in self.options:
                    col1 = ':%s:`%s <%s>`\\ %s' % (qualifier, name, real_name, rst.escape(sig))
                else:
                    col1 = ':%s:`%s <%s>`' % (qualifier, name, real_name)
                col2 = summary
                col3 = env_sum
                col4 = env_warn
                append_row(col1, col2, col3, col4)

        return [table_spec, table]

class MsPlatWarnAutoSummary(MsAutosummary):
    """
    Inherited from MsAutosummary. Add a third column about Note` to the table
    and Add a fourth column about `Supported Platforms` to the table.
    """
    def init(self):
        """
        init method
        """
        self.find_doc_name_fourth = ".. warning::"
        self.fourth_title = "**Warning**"
        self.default_doc_fourth = "None"
        self.find_doc_name = "Supported Platforms:"
        self.third_title = "**{}**".format(self.find_doc_name[:-1])
        self.default_doc = ""

    def get_refer_platform(self, name=None):
        """Get the `Supported Platforms`."""
        if not name:
            return []
        try:
            api_obj = get_api(name)
            if not api_obj:
                logger.warning(f"not api obj: {name}")
                return ""
            api_doc = inspect.getdoc(api_obj)
            if '.ops.' in name and 'Refer to' in api_doc.split('\n')[-1]:
                new_name = re.findall(r'Refer to :\w+:`(.*?)` for more details.', api_doc.split('\n')[-1])[0]
                api_doc = inspect.getdoc(get_api(new_name))
                platform_str = re.findall(r'Supported Platforms:\n\s+(.*?)\n\n', api_doc)
                if not platform_str:
                    platform_str_leak = re.findall(r'Supported Platforms:\n\s+(.*)', api_doc)
                    if platform_str_leak:
                        return platform_str_leak[0]
                    logger.warning(f"not find Supported Platforms: {name}")
                    return ""
                return platform_str[0]
            return ""
        except: #pylint: disable=bare-except
            return ""

class MsNoteAutoSummary(MsAutosummary):
    """
    Inherited from MsAutosummary. Add a third column about `Note` to the table.
    """

    def init(self):
        """
        init method
        """
        self.find_doc_name = ".. note::"
        self.third_title = "**Note**"
        self.default_doc = "None"
        self.find_doc_name_fourth = ""
        self.fourth_title = ""
        self.default_doc_fourth = ""

    def extract_env_summary(self, doc: List[str]) -> str:
        """Extract env summary from docstring."""
        env_sum = self.default_doc
        for piece in doc:
            if piece.startswith(self.find_doc_name):
                env_sum = piece[10:]
        return env_sum

class MsPlatformAutoSummary(MsAutosummary):
    """
    Inherited from MsAutosummary. Add a third column about `Supported Platforms` to the table.
    """
    def init(self):
        """
        init method
        """
        self.find_doc_name = "Supported Platforms:"
        self.third_title = "**{}**".format(self.find_doc_name[:-1])
        self.default_doc = ""
        self.find_doc_name_fourth = ""
        self.fourth_title = ""
        self.default_doc_fourth = ""

    def get_refer_platform(self, name=None):
        """Get the `Supported Platforms`."""
        if not name:
            return []
        try:
            api_obj = get_api(name)
            if not api_obj:
                logger.warning(f"not api obj: {name}")
                return ""
            api_doc = inspect.getdoc(api_obj)
            if '.ops.' in name and 'Refer to' in api_doc.split('\n')[-1]:
                new_name = re.findall(r'Refer to :\w+:`(.*?)` for more details.', api_doc.split('\n')[-1])[0]
                api_doc = inspect.getdoc(get_api(new_name))
                platform_str = re.findall(r'Supported Platforms:\n\s+(.*?)\n\n', api_doc)
                if not platform_str:
                    platform_str_leak = re.findall(r'Supported Platforms:\n\s+(.*)', api_doc)
                    if platform_str_leak:
                        return platform_str_leak[0]
                    logger.warning(f"not find Supported Platforms: {name}")
                    return ""
                return platform_str[0]
            return ""
        except: #pylint: disable=bare-except
            return ""

class MsCnAutoSummary(Autosummary):
    """Overwrite MsPlatformAutosummary for chinese python api."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_head = ()
        self.find_doc_name = ""
        self.third_title = ""
        self.default_doc = ""
        self.third_name_en = ""
        self.fourth_name_en = ""
        self.default_doc_fourth = ""

    def get_third_column_en(self, doc):
        """Get the third column for en."""
        third_column = self.default_doc
        for i, piece in enumerate(doc):
            if piece.startswith(self.third_name_en):
                try:
                    if "eprecated" in doc[i+1][4:]:
                        third_column = "弃用"
                    else:
                        third_column = doc[i+1][4:]
                except IndexError:
                    third_column = ''
        return third_column

    def get_fourth_column_en(self, doc):
        """Get the fourth column for en."""
        flag_warn = 0
        env_warn = ''
        for piece in doc:
            if flag_warn and piece != '':
                if piece.lstrip().startswith('- '):
                    env_warn += piece.lstrip().lstrip('- ')+' '
                else:
                    env_warn += piece.lstrip()+' '
            elif flag_warn and piece == '':
                break
            if piece.startswith(self.fourth_name_en):
                flag_warn = 1
        if not env_warn:
            env_warn = self.default_doc_fourth
        return env_warn.rstrip()

    # def get_summary_re(self, display_name: str):
    #     return re.compile(rf'\.\. \w+:\w+::\s+{display_name}.*?\n\n\s+((?:.|\n|)+?)\n\n')

    def run(self) -> List[Node]:
        self.bridge = DocumenterBridge(self.env, self.state.document.reporter,
                                       Options(), self.lineno, self.state)

        names = [x.strip().split()[0] for x in self.content
                 if x.strip() and re.search(r'^[~a-zA-Z_]', x.strip()[0])]
        items = self.get_items(names)
        #pylint: disable=redefined-outer-name
        nodes = self.get_table(items)

        dirname = posixpath.dirname(self.env.docname)

        tree_prefix = self.options['toctree'].strip()
        docnames = []
        names = [i[0] for i in items]
        for name in names:
            docname = posixpath.join(tree_prefix, name)
            docname = posixpath.normpath(posixpath.join(dirname, docname))
            if docname not in self.env.found_docs:
                continue

            docnames.append(docname)

        if docnames:
            tocnode = addnodes.toctree()
            tocnode['includefiles'] = docnames
            tocnode['entries'] = [(None, docn) for docn in docnames]
            tocnode['maxdepth'] = -1
            tocnode['glob'] = None

            nodes.append(autosummary_toc('', '', tocnode))

        return nodes

    def get_items(self, names: List[str]) -> List[Tuple[str, str, str, str]]:
        """Try to import the given names, and return a list of
        ``[(name, signature, summary_string, real_name), ...]``.
        """
        prefixes = get_import_prefixes_from_env(self.env)
        doc_path = os.path.dirname(self.state.document.current_source)
        items = []  # type: List[Tuple[str, str, str, str]]
        max_item_chars = 50
        origin_rst_files = self.env.config.rst_files
        all_rst_files = self.env.found_docs
        generated_files = all_rst_files.difference(origin_rst_files)

        for name in names:
            display_name = name
            if name.startswith('~'):
                name = name[1:]
                display_name = name.split('.')[-1]

            dir_name = self.options['toctree']
            file_path = os.path.join(doc_path, dir_name, display_name+'.rst')
            spec_path = os.path.join('api_python', dir_name, display_name)
            if os.path.exists(file_path) and spec_path not in generated_files:
                summary_re_tag = re.compile(
                    rf'\.\. \w+:\w+::\s+{display_name}.*?\n\s+:.*?:\n\n\s+((?:.|\n|)+?)(\n\n|。)')
                summary_re_wrap = re.compile(rf'\.\. \w+:\w+::\s+{display_name}(?:.|\n|)+?\n\n\s+((?:.|\n|)+?)(\n\n|。)')
                summary_re = re.compile(rf'\.\. \w+:\w+::\s+{display_name}.*?\n\n\s+((?:.|\n|)+?)(\n\n|。)')
                content = ''
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content:
                    content = re.sub('\n[ ]+\n', '\n\n', content)
                    summary_str = summary_re.findall(content)
                    summary_str_tag = summary_re_tag.findall(content)
                    summary_str_wrap = summary_re_wrap.findall(content)
                    if '.ops.' in display_name and '更多详情请查看：' in content.split('\n')[-2]:
                        summary_str = content.split('\n')[-2].strip()
                    elif '.mint.' in display_name and '更多详情请查看：' in content.split('\n')[-2]:
                        summary_str = content.split('\n')[-2].strip()
                    elif summary_str_tag:
                        if re.findall("[:：,，。.;；]", summary_str_tag[0][0][-1]):
                            logger.warning(f"{display_name}接口的概述格式需调整")
                        summary_str = re.sub('\n[ ]+', '', summary_str_tag[0][0]) + '。'
                    elif summary_str:
                        if re.findall("[:：,，。.;；]", summary_str[0][0][-1]):
                            logger.warning(f"{display_name}接口的概述格式需调整")
                        summary_str = re.sub('\n[ ]+', '', summary_str[0][0]) + '。'
                    elif summary_str_wrap:
                        if re.findall("[:：,，。.;；]", summary_str_wrap[0][0][-1]):
                            logger.warning(f"{display_name}接口的概述格式需调整")
                        summary_str = re.sub('\n[ ]+', '', summary_str_wrap[0][0]) + '。'
                    else:
                        summary_str = ''
                    if not self.table_head:
                        items.append((display_name, summary_str))
                    elif len(self.table_head) == 3:
                        third_str = self.get_third_column(display_name, content)
                        if third_str:
                            third_str = third_str[0]
                        else:
                            third_str = ''
                        items.append((display_name, summary_str, third_str))
                    elif len(self.table_head) == 4:
                        third_str = self.get_third_column(display_name, content)
                        if third_str:
                            third_str = third_str[0]
                        else:
                            third_str = ''
                        fourth_str = self.get_fourth_column(display_name, content)
                        if not fourth_str:
                            fourth_str = '无'
                        items.append((display_name, summary_str, third_str, fourth_str))
                    elif len(self.table_head) == 5:
                        third_str = self.get_third_column(display_name, content)
                        if third_str:
                            third_str = third_str[0]
                        else:
                            third_str = ''
                        fourth_str = self.get_fourth_column(display_name, content)
                        if not fourth_str:
                            fourth_str = '无'
                        fifth_str = self.get_fifth_column(display_name)
                        if fifth_str.endswith(', '):
                            fifth_str = fifth_str[:-2]
                        items.append((display_name, summary_str, third_str, fourth_str, fifth_str))
            else:
                try:
                    with mock(self.config.autosummary_mock_imports):
                        real_name, obj, parent, modname = import_by_name(name, prefixes=prefixes)
                except ImportError:
                    logger.warning(__('failed to import %s'), name)
                    if len(self.table_head) == 3:
                        items.append((name, '', ''))
                    elif len(self.table_head) == 4:
                        items.append((name, '', '', ''))
                    continue

                self.bridge.result = StringList()  # initialize for each documenter
                full_name = real_name
                if not isinstance(obj, ModuleType):
                    # give explicitly separated module name, so that members
                    # of inner classes can be documented
                    full_name = modname + '::' + full_name[len(modname) + 1:]
                # NB. using full_name here is important, since Documenters
                #     handle module prefixes slightly differently
                doccls = get_documenter(self.env.app, obj, parent)
                documenter = doccls(self.bridge, full_name)

                if not documenter.parse_name():
                    logger.warning(__('failed to parse name %s'), real_name)
                    if len(self.table_head) == 3:
                        items.append((display_name, '', ''))
                    elif len(self.table_head) == 4:
                        items.append((display_name, '', '', ''))
                    continue
                if not documenter.import_object():
                    logger.warning(__('failed to import object %s'), real_name)
                    if len(self.table_head) == 3:
                        items.append((display_name, '', ''))
                    elif len(self.table_head) == 4:
                        items.append((display_name, '', '', ''))
                    continue
                if documenter.options.members and not documenter.check_module():
                    continue

                # try to also get a source code analyzer for attribute docs
                try:
                    documenter.analyzer = ModuleAnalyzer.for_module(
                        documenter.get_real_modname())
                    # parse right now, to get PycodeErrors on parsing (results will
                    # be cached anyway)
                    documenter.analyzer.find_attr_docs()
                except PycodeError as err:
                    logger.debug('[autodoc] module analyzer failed: %s', err)
                    # no source file -- e.g. for builtin and C modules
                    documenter.analyzer = None

                # -- Grab the signature

                try:
                    sig = documenter.format_signature(show_annotation=False)
                except TypeError:
                    # the documenter does not support ``show_annotation`` option
                    sig = documenter.format_signature()

                if not sig:
                    sig = ''
                else:
                    max_chars = max(10, max_item_chars - len(display_name))
                    sig = mangle_signature(sig, max_chars=max_chars)

                # -- Grab the summary and third_colum

                documenter.add_content(None)
                summary = extract_summary(self.bridge.result.data[:], self.state.document)
                if self.table_head:
                    third_colum = self.get_third_column_en(self.bridge.result.data[:])
                    if len(self.table_head) == 3:
                        items.append((display_name, summary, third_colum))
                    elif len(self.table_head) == 4:
                        fourth_colum = self.get_fourth_column_en(self.bridge.result.data[:])
                        items.append((display_name, summary, third_colum, fourth_colum))
                else:
                    items.append((display_name, summary))


        return items

    def get_table(self, items: List[Tuple[str, str, str]]) -> List[Node]:
        """Generate a proper list of table nodes for autosummary:: directive.

        *items* is a list produced by :meth:`get_items`.
        """
        table_spec = addnodes.tabular_col_spec()
        table = autosummary_table('')
        real_table = nodes.table('', classes=['longtable'])
        table.append(real_table)

        if not self.table_head:
            table_spec['spec'] = r'\X{1}{2}\X{1}{2}'
            group = nodes.tgroup('', cols=2)
            real_table.append(group)
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=90))
        elif len(self.table_head) == 3:
            table_spec['spec'] = r'\X{2}{5}\X{2}{5}\X{1}{5}'
            group = nodes.tgroup('', cols=3)
            real_table.append(group)
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=60))
            group.append(nodes.colspec('', colwidth=30))
        elif len(self.table_head) == 4:
            table_spec['spec'] = r'\X{1}{3}\X{1}{3}\X{1}{6}\X{1}{6}'
            group = nodes.tgroup('', cols=4)
            real_table.append(group)
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=45))
            group.append(nodes.colspec('', colwidth=25))
            group.append(nodes.colspec('', colwidth=20))
        elif len(self.table_head) == 5:
            table_spec['spec'] = r'\X{1}{4}\X{1}{3}\X{1}{12}\X{2}{12}\X{2}{12}'
            group = nodes.tgroup('', cols=5)
            real_table.append(group)
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=35))
            group.append(nodes.colspec('', colwidth=10))
            group.append(nodes.colspec('', colwidth=14))
            group.append(nodes.colspec('', colwidth=31))

        body = nodes.tbody('')
        group.append(body)

        def append_row(*column_texts: str) -> None:
            row = nodes.row('')
            source, line = self.state_machine.get_source_and_line()
            for text in column_texts:
                node = nodes.paragraph('')
                vl = StringList()
                vl.append(text, '%s:%d:<autosummary>' % (source, line))
                with switch_source_input(self.state, vl):
                    self.state.nested_parse(vl, 0, node)
                    try:
                        if isinstance(node[0], nodes.paragraph):
                            node = node[0]
                    except IndexError:
                        pass
                    row.append(nodes.entry('', node))
            body.append(row)
        append_row(*self.table_head)
        if not self.table_head:
            for name, summary in items:
                qualifier = 'obj'
                col1 = ':%s:`%s <%s>`' % (qualifier, name, name)
                col2 = summary
                append_row(col1, col2)
        elif len(self.table_head) == 3:
            for name, summary, other in items:
                qualifier = 'obj'
                col1 = ':%s:`%s <%s>`' % (qualifier, name, name)
                col2 = summary
                col3 = other
                append_row(col1, col2, col3)
        elif len(self.table_head) == 4:
            for name, summary, other, other1 in items:
                qualifier = 'obj'
                col1 = ':%s:`%s <%s>`' % (qualifier, name, name)
                col2 = summary
                col3 = other
                col4 = other1
                append_row(col1, col2, col3, col4)
        elif len(self.table_head) == 5:
            for name, summary, other, other1, other2 in items:
                qualifier = 'obj'
                col1 = ':%s:`%s <%s>`' % (qualifier, name, name)
                col2 = summary
                col3 = other
                col4 = other1
                col5 = other2
                append_row(col1, col2, col3, col4, col5)
        return [table_spec, table]

def get_api(fullname):
    """Get the api module."""
    try:
        module_name, api_name = ".".join(fullname.split('.')[:-1]), fullname.split('.')[-1]
        # pylint: disable=unused-variable
        module_import = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name, api_name = ".".join(fullname.split('.')[:-2]), ".".join(fullname.split('.')[-2:])
        module_import = importlib.import_module(module_name)
    # pylint: disable=eval-used
    api = getattr(module_import, api_name, '')
    return api

class MsCnPlatformAutoSummary(MsCnAutoSummary):
    """definition of mscnplatformautosummary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_head = ('**接口名**', '**概述**', '**支持平台**')
        self.third_name_en = "Supported Platforms:"

    def get_third_column(self, name=None, content=None):
        """Get the`Supported Platforms`."""
        if not name:
            return []
        try:
            if '.mint.' in name:
                return ["``Ascend``"]
            api_obj = get_api(name)
            if not api_obj:
                logger.warning(f"not api obj: {name}")
                return []
            api_doc = inspect.getdoc(api_obj)
            if '.ops.' in name and 'Refer to' in api_doc.split('\n')[-1]:
                new_name = re.findall(r'Refer to :\w+:`(.*?)` for more details.', api_doc.split('\n')[-1])[0]
                api_doc = inspect.getdoc(get_api(new_name))
            platform_str = re.findall(r'Supported Platforms:\n\s+(.*?)\n\n', api_doc)
            if not platform_str:
                platform_str = re.findall(r'Supported Platforms:\n\s+(.*)', api_doc)
            if platform_str in (['deprecated'], ['Deprecated']):
                return ["弃用"]
            if not platform_str:
                logger.warning(f"not find Supported Platforms: {name}")
                return []
            return platform_str
        except: #pylint: disable=bare-except
            return []

class MsCnPlatWarnAutoSummary(MsCnAutoSummary):
    """definition of mscnplatwarnautosummary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_head = ('**接口名**', '**概述**', '**支持平台**', '**警告**')
        self.third_name_en = "Supported Platforms:"
        self.fourth_name_en = ".. warning::"
        self.default_doc_fourth = "None"

    def get_third_column(self, name=None, content=None):
        """Get the `Supported Platforms`."""
        if not name:
            return []
        try:
            if '.mint.' in name:
                return ["``Ascend``"]
            api_obj = get_api(name)
            if not api_obj:
                logger.warning(f"not api obj: {name}")
                return []
            api_doc = inspect.getdoc(api_obj)
            if '.ops.' in name and 'Refer to' in api_doc.split('\n')[-1]:
                new_name = re.findall(r'Refer to :\w+:`(.*?)` for more details.', api_doc.split('\n')[-1])[0]
                api_doc = inspect.getdoc(get_api(new_name))
            platform_str = re.findall(r'Supported Platforms:\n\s+(.*?)\n\n', api_doc)
            if not platform_str:
                platform_str = re.findall(r'Supported Platforms:\n\s+(.*)', api_doc)
            if platform_str in (['deprecated'], ['Deprecated']):
                return ["弃用"]
            if not platform_str:
                logger.warning(f"not find Supported Platforms: {name}")
                return []
            return platform_str
        except: #pylint: disable=bare-except
            return []

    def get_fourth_column(self, name=None, content=''):
        """Get the `Warning`."""
        env_warn = ''
        fourth_str = ''
        try:
            if name.split('.')[-1][0].isupper() and '.. py:method::' in content:
                content = content.split('.. py:method::')[0]
            if re.findall(r'\.\. warning::\n', content):
                indent = re.findall(r'([ ]+)\.\. warning::\n', content)[0]
                if re.findall(rf'\.\. warning::\n((?:.|\n|)+?)\n\n{indent}\S', content):
                    env_warn = re.findall(rf'\.\. warning::\n((?:.|\n|)+?)\n\n{indent}\S', content)[0]
                elif re.findall(rf'\.\. warning::\n((?:.|\n|)+)\n', content):
                    env_warn = re.findall(rf'\.\. warning::\n((?:.|\n|)+)\n', content)[0]
                for line in env_warn.split('\n'):
                    if line == '':
                        continue
                    elif '.. include::' in line:
                        break
                    elif len(line) - len(line.lstrip()) <= len(indent):
                        logger.warning(name + ' Note has error format')
                        break
                    else:
                        if line.lstrip().startswith('- '):
                            fourth_str += line.lstrip().lstrip('- ')+' '
                        else:
                            fourth_str += line.lstrip()+' '
        except IndexError:
            logger.warning(name + 'get Warning error')
        return fourth_str

class MsCnPlataclnnAutoSummary(MsCnAutoSummary):
    """definition of mscnplatwarnautosummary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_head = ('**接口名**', '**概述**', '**支持平台**', '**警告**', '**aclnn算子**')
        self.third_name_en = "Supported Platforms:"
        self.fourth_name_en = ".. warning::"
        self.default_doc_fourth = "None"
        self.fifth_name_en = ""

    def get_third_column(self, name=None, content=None):
        """Get the `Supported Platforms`."""
        if not name:
            return []
        try:
            if '.mint.' in name:
                return ["``Ascend``"]
            api_obj = get_api(name)
            if not api_obj:
                logger.warning(f"not api obj: {name}")
                return []
            api_doc = inspect.getdoc(api_obj)
            if '.ops.' in name and 'Refer to' in api_doc.split('\n')[-1]:
                new_name = re.findall(r'Refer to :\w+:`(.*?)` for more details.', api_doc.split('\n')[-1])[0]
                api_doc = inspect.getdoc(get_api(new_name))
            platform_str = re.findall(r'Supported Platforms:\n\s+(.*?)\n\n', api_doc)
            if not platform_str:
                platform_str = re.findall(r'Supported Platforms:\n\s+(.*)', api_doc)
            if platform_str in (['deprecated'], ['Deprecated']):
                return ["弃用"]
            if not platform_str:
                logger.warning(f"not find Supported Platforms: {name}")
                return []
            return platform_str
        except: #pylint: disable=bare-except
            return []

    def get_fourth_column(self, name=None, content=''):
        """Get the `Warning`."""
        env_warn = ''
        fourth_str = ''
        try:
            if name.split('.')[-1][0].isupper() and '.. py:method::' in content:
                content = content.split('.. py:method::')[0]
            if re.findall(r'\.\. warning::\n', content):
                indent = re.findall(r'([ ]+)\.\. warning::\n', content)[0]
                if re.findall(rf'\.\. warning::\n((?:.|\n|)+?)\n\n{indent}\S', content):
                    env_warn = re.findall(rf'\.\. warning::\n((?:.|\n|)+?)\n\n{indent}\S', content)[0]
                elif re.findall(rf'\.\. warning::\n((?:.|\n|)+)\n', content):
                    env_warn = re.findall(rf'\.\. warning::\n((?:.|\n|)+)\n', content)[0]
                for line in env_warn.split('\n'):
                    if line == '':
                        continue
                    elif '.. include::' in line:
                        break
                    elif len(line) - len(line.lstrip()) <= len(indent):
                        logger.warning(name + ' Note has error format')
                        break
                    else:
                        if line.lstrip().startswith('- '):
                            fourth_str += line.lstrip().lstrip('- ')+' '
                        else:
                            fourth_str += line.lstrip()+' '
        except IndexError:
            logger.warning(name + 'get Warning error')
        return fourth_str

    def get_fifth_column(self, name=None):
        """Get the mint mapping aclnn."""
        if name in self.env.config.mint_aclnn:
            return self.env.config.mint_aclnn[name]
        print(f'no_aclnn_url: {name}')
        return '无'

class MsCnNoteAutoSummary(MsCnAutoSummary):
    """definition of cnmsnoteautosummary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_head = ('**接口名**', '**概述**', '**说明**')
        self.third_name_en = ".. note::"

    def get_third_column(self, name=None, content=''):
        note_re = re.compile(r'\.\. note::\n{,2}\s+(.*?)[。\n]')
        third_str = note_re.findall(content)
        return third_str
