"""自定义锚点生成扩展
用于恢复 nodes.txt 中的锚点生成逻辑，避免 Sphinx 7.4.7 原生 set_id 生成的锚点格式（如 id1、id2） 。
在 conf.py 的 extensions 中添加 'custom_anchor' 即可启用。
"""
import re
from docutils import nodes

def short_title(titlename):
    """把标题转换成锚点格式"""
    titlename = titlename.lower()
    cn_symbol = "。；，“”（）、？《》"
    en_symbol = ".,=+*()<>[]{}|/&"
    spec_symbol = "_:："
    titlename = titlename.replace("==", "").replace("!=", "")
    for i in cn_symbol:
        titlename = titlename.replace(i, '')
    for j in en_symbol:
        titlename = titlename.replace(j, '')
    for k in spec_symbol:
        titlename = titlename.replace(k, ' ')
    titlename = titlename.replace('  ', ' ').replace(' ', '-').replace('"', '').replace("'", "")

    return titlename

def custom_set_id(self, node, msgnode=None, suggested_prefix=''):
    """恢复 nodes.txt 中的锚点生成逻辑"""
    if node['ids']:
        # register and check for duplicates
        for node_id in node['ids']:
            self.ids.setdefault(node_id, node)
            if self.ids[node_id] is not node:
                msg = self.reporter.severe(f'Duplicate ID: "{node_id}".')
                if msgnode is not None:
                    msgnode += msg
        return node['ids'][-1]
    # generate and set id
    id_prefix = self.settings.id_prefix
    auto_id_prefix = self.settings.auto_id_prefix
    base_id = ''
    node_id = ''
    for name in node['names']:
        base_id = nodes.make_id(name)
        node_id = id_prefix + base_id
        # TODO: allow names starting with numbers if `id_prefix`
        # is non-empty:  id = make_id(id_prefix + name)
        if base_id and node_id not in self.ids:
            break
    else:
        if base_id and auto_id_prefix.endswith('%'):
            # disambiguate name-derived ID
            # TODO: remove second condition after announcing change
            prefix = node_id + '-'
        else:
            prefix = id_prefix + auto_id_prefix
            if prefix.endswith('%'):
                prefix = f'{prefix[:-1]}{suggested_prefix or nodes.make_id(node.tagname)}-'
        while True:
            self.id_counter[prefix] += 1
            node_id = f'{prefix}{self.id_counter[prefix]}'
            if node_id not in self.ids:
                break

    flag = 0
    if node['names']:
        node_id = node['names'][0]
        origin_id = node_id
        node_id = short_title(node_id)
        self.id_counter[node_id] += 1
        if self.id_counter[node_id] > 1:
            node_id = f'{node_id}-{self.id_counter[node_id]-1}'
            self.id_counter[node_id] += 1
        else:
            zhcn_pattern = re.findall(r'[\u4e00-\u9fa5]', origin_id)
            if not zhcn_pattern and origin_id == origin_id.lower():
                flag = 1
    if flag == 1:
        rep_symbol = "._:"
        empty_symbol = "()&"
        for s in rep_symbol:
            origin_id = origin_id.replace(s, '-')
        for s in empty_symbol:
            origin_id = origin_id.replace(s, '')
        node['ids'].append(
            origin_id.replace('"', '').replace(" ", "-").replace("--", "-")
            .replace("==", "").replace("!=", "").replace("=", "")
        )
    else:
        node['ids'].append(node_id)
    self.ids[node_id] = node
    return node_id

def setup(app):
    def inject_anchor(builder):
        del builder
        nodes.short_title = short_title
        nodes.document.set_id = custom_set_id
        print("custom_anchor: 已注入自定义锚点逻辑")

    app.connect('builder-inited', inject_anchor)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
