"""sphinx replace"""
import os
import sphinx
sphinx_dir = os.path.dirname(os.path.dirname(sphinx.__file__))

with open(os.path.join(sphinx_dir, 'sphinx/util/docutils.py'), 'r+', encoding='utf-8') as f:
    content = f.read()
    old_content = '# cache a vanilla instance of nodes.document'

    new_content = """from packaging import version
from docutils.nodes import Node
__version_info__ = version.parse(docutils.__version__).release
if __version_info__ < (0, 21):
    def findall(self, *args, **kwargs):
        return iter(self.traverse(*args, **kwargs))
Node.findall = findall #type: ignore
"""
    content = content.replace(old_content, new_content)
    f.seek(0)
    f.truncate()
    f.write(content)

with open(os.path.join(sphinx_dir, 'sphinx/search/__init__.py'), 'r+', encoding='utf-8') as f:
    content = f.read()
    old_content = """elif (isinstance(node, nodes.meta) # type: ignore[attr-defined]
                  and _is_meta_keywords(node, language)):
                keywords = [keyword.strip() for keyword in node['content'].split(',')]
                word_store.words.extend(keywords)"""

    new_content = ''
    content = content.replace(old_content, new_content)
    f.seek(0)
    f.truncate()
    f.write(content)

with open(os.path.join(sphinx_dir, 'sphinx/addnodes.py'), 'r+', encoding='utf-8') as f:
    content = f.read()
    old_content = """# deprecated name -> (object to return, canonical path or empty string)
_DEPRECATED_OBJECTS = {
    'meta': (nodes.meta, 'docutils.nodes.meta'),  # type: ignore[attr-defined]
    'docutils_meta': (nodes.meta, 'docutils.nodes.meta'),  # type: ignore[attr-defined]
}


def __getattr__(name):
    if name not in _DEPRECATED_OBJECTS:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)

    from sphinx.deprecation import _deprecation_warning

    deprecated_object, canonical_name = _DEPRECATED_OBJECTS[name]
    _deprecation_warning(__name__, name, canonical_name, remove=(7, 0))
    return deprecated_object"""
    new_content = ''
    content = content.replace(old_content, new_content)
    f.seek(0)
    f.truncate()
    f.write(content)

with open(os.path.join(sphinx_dir, 'sphinx/util/__init__.py'), 'r+', encoding='utf-8') as f:
    content = f.read()
    old_content = '# a regex to recognize coding cookies'
    new_content = """def get_full_modname(modname: str, attribute: str) -> str | None:
    if modname is None:
        # Prevents a TypeError: if the last getattr() call will return None
        # then it's better to return it directly
        return None
    module = import_module(modname)

    # Allow an attribute to have multiple parts and incidentally allow
    # repeated .s in the attribute.
    value = module
    for attr in attribute.split('.'):
        if attr:
            value = getattr(value, attr)

    return getattr(value, '__module__', None)
    """
    content = content.replace(old_content, new_content)
    f.seek(0)
    f.truncate()
    f.write(content)

with open(os.path.join(sphinx_dir, 'docutils/transforms/frontmatter.py'), 'r+', encoding='utf-8') as f:
    content = f.read()
    content = content.replace('nodes.meta', 'nodes.document')
    f.seek(0)
    f.truncate()
    f.write(content)
