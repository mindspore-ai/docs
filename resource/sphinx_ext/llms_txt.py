"""自定义 llms.txt / llms-full.txt 生成器，基于已解析的 doctree。"""
import os
from sphinx.util import logging
from sphinx_markdown_builder.translator import MarkdownTranslator

logger = logging.getLogger(__name__)

class _MarkdownConfig:
    """Config proxy with default values for sphinx-markdown-builder's translator."""

    markdown_http_base = ""
    markdown_uri_doc_suffix = ".html"
    markdown_anchor_sections = False
    markdown_anchor_signatures = False
    markdown_docinfo = False
    markdown_bullet = "*"
    markdown_flavor = ""


class _MarkdownBuilderProxy:
    """Minimal proxy providing the attributes MarkdownTranslator needs from a builder."""

    def __init__(self, current_doc_name: str = ""):
        self.config = _MarkdownConfig()
        self.current_doc_name = current_doc_name

    def get_target_uri(self, docname: str, typ: str = None) -> str:  # pylint: disable=unused-argument
        return f"{docname}.html"


def _doctree_to_markdown(doctree, docname):
    """使用 sphinx-markdown-builder 提供的 translator，把 Sphinx 的 doctree 转换成 Markdown。"""

    proxy = _MarkdownBuilderProxy(current_doc_name=docname)
    visitor = MarkdownTranslator(doctree, proxy)
    doctree.walkabout(visitor)
    return visitor.astext()

def _collect_docnames(app):
    """按 toctree 顺序收集所有 docname。"""
    docnames = []
    visited = set()

    def walk(name):
        if name in visited:
            return
        visited.add(name)
        docnames.append(name)
        for child in app.env.toctree_includes.get(name, []):
            walk(child)

    walk(app.env.config.master_doc)

    #补上不在 toctree 里的孤立页面
    for name in sorted(app.env.all_docs):
        if name not in visited:
            docnames.append(name)

    return docnames

def _generate_llms_full_txt(app):
    """生成 llms-full.txt ，内容来自 doctree 的 Markdown 转换。"""
    project = app.config.project
    parts = [f"# {project}\n"]

    for docname in _collect_docnames(app):
        doctree = None
        try:
            doctree = app.env.get_doctree(docname)
            md = _doctree_to_markdown(doctree, docname)
        except (ValueError, TypeError, OSError) as e:
            logger.warning(
                "llms_txt: failed to convert %s: %s: %s",
                docname, type(e).__name__, e,
            )
            if doctree is not None:
                text = doctree.astext()
                title = app.env.titles.get(docname)
                title_str = title.astext() if title else docname
                md = f"# {title_str}\n\n{text}"
            else:
                md = f"# {docname}\n\n(conversion failed)"

        parts.append(f"\n---\n\n{md}")

        # 同时写一份单页 .md，供 llms.txt 的相对链接使用
        page_md_path = os.path.join(app.outdir, f"{docname}.md")
        os.makedirs(os.path.dirname(page_md_path), exist_ok=True)
        with open(page_md_path, "w", encoding="utf-8") as f:
            f.write(md)

    content = "\n".join(parts) + "\n"
    outpath = os.path.join(app.outdir, "llms-full.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("llms_txt: wrote %s and per-page .md files", outpath)

def _generate_llms_txt(app):
    """生成 llms.txt 摘要索引，Markdown 格式的站点地图。"""
    lines = [f"# {app.config.project}\n"]

    for docname in _collect_docnames(app):
        title = app.env.titles.get(docname)
        title_str = title.astext() if title else docname
        lines.append(f"- [{title_str}]({docname}.md)")

    content = "\n".join(lines) + "\n"
    outpath = os.path.join(app.outdir, "llms.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("llms_txt: wrote %s", outpath)


def setup(app):
    app.connect("build-finished", lambda app, exc: _generate_llms_full_txt(app))
    app.connect("build-finished", lambda app, exc: _generate_llms_txt(app))
    return {"version": "0.1", "parallel_read_safe": True}
