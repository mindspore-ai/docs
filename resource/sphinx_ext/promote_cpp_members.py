"""Promote C++ members to class level for Sphinx documentation."""

from docutils import nodes
from sphinx import addnodes

def promote_cpp_members(_, doctree, docname):
    """Promote C++ function descriptions into individual sections for TOC indexing.

    Only processes pages under ``generate/class``. Each ``cpp:function`` desc
    node is wrapped in a ``nodes.section`` with an implicit target so that
    Sphinx's TOC builder can pick it up.
    """
    del _  # unused, required by Sphinx extension interface
    # 只处理 generate/class 开头的页面
    if not docname.startswith('generate/class'):
        return

    # 找到所有 cpp function 的 desc 节点
    used = set()
    for desc in list(doctree.findall(addnodes.desc)):
        if desc.get('domain') != 'cpp' or desc.get('objtype') != 'function':
            continue

        # 从 desc_signature 里拿函数名
        sig = desc.next_node(addnodes.desc_signature)
        if sig is None:
            continue
        name_node = sig.next_node(addnodes.desc_name)
        if name_node is None:
            continue
        name = name_node.astext()
        if not name:
            continue

        # 创建 section
        base = nodes.make_id(name)
        sec_id = base
        i = 1
        while sec_id in used:
            sec_id = f'{base}-{i}'
            i += 1
        used.add(sec_id)
        sec = nodes.section('', ids=[sec_id], names=[sec_id])
        sec += nodes.title('', name)

        # 把 desc 从原位置移出，放进 section
        parent = desc.parent
        idx = parent.index(desc)
        parent.remove(desc)
        sec += desc
        parent.insert(idx, sec)

        # 关键: 注册 section，让 TOC 能识别
        doctree.note_implicit_target(sec)

def setup(app):
    """Register the extension with Sphinx."""
    app.connect('doctree-resolved', promote_cpp_members)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
