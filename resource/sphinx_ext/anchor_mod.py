"""Modify docutils.nodes."""
import inspect
from docutils import nodes as docu_node
from docutils.nodes import document


short_title_define = '''\
def short_title(titlename):
    titlename = titlename.lower()
    cn_symbol = "。；，“”（）、？《》"
    en_symbol = ".,=+*()<>[]{}|/&"
    spec_symbol = "_:："
    for i in cn_symbol:
        titlename = titlename.replace(i, '')
    for j in en_symbol:
        titlename = titlename.replace(j, '')
    for k in spec_symbol:
        titlename = titlename.replace(k, ' ')
    titlename = titlename.replace('  ', ' ').replace(' ', '-')

    return titlename
'''

dupname_rewrite = """
def dupname(node, name):
    node['dupnames'].append(name)
    if node['names']:
        node['names'].remove(name)
    node.referenced = 1
"""

before = "            node['ids'].append(id)"
after = """\
        import re
        flag = 0
        if node['names']:
            id = node['names'][0]
            origin_id = id
            id = short_title(id)
            self.id_counter[id] += 1
            if self.id_counter[id] > 1:
                id = '{}-{}'.format(id, self.id_counter[id]-1)
                self.id_counter[id] += 1
            else:
                zhcnPattern = re.findall(r'[\u4e00-\u9fa5]',origin_id)
                if not zhcnPattern and origin_id==origin_id.lower():
                    flag = 1
        if flag==1:
            node['ids'].append(origin_id.replace(".","-").replace("_","-").replace("::","-").replace(" ","-"))
        else:
            node['ids'].append(id)"""

# Mod nodes for docutils.
document_content = inspect.getsource(document)
document_content = document_content.replace(before, after)
#pylint: disable=exec-used
exec(short_title_define, docu_node.__dict__)
exec(document_content, docu_node.__dict__)
exec(dupname_rewrite, docu_node.__dict__)
