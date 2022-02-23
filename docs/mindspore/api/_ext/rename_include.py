"""Rename .rst file to .txt file for include directive."""
import os
import re
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

origin = "rst"
replace = "txt"

include_re = re.compile(r'\.\. include::\s+(.*?)(\.rst|\.txt)')
include_re_sub = re.compile(rf'(\.\. include::\s+(.*?))\.{origin}')

def rename_include(api_dir):
    """
    Rename .rst file to .txt file for include directive.

    api_dir - api path relative.
    """
    tar = []
    for root, _, files in os.walk(api_dir):
        for file in files:
            if not file.endswith('.rst'):
                continue
            try:
                with open(os.path.join(root, file), 'r+', encoding='utf-8') as f:
                    content = f.read()
                    tar_ = include_re.findall(content)
                    if tar_:
                        tar_ = [i[0].split('/')[-1]+f'.{origin}' for i in tar_]
                        tar.extend(tar_)
                        sub = include_re_sub.findall(content)
                        if sub:
                            content_ = include_re_sub.sub(rf'\1.{replace}', content)
                            f.seek(0)
                            f.truncate()
                            f.write(content_)
            except UnicodeDecodeError:
                # pylint: disable=logging-fstring-interpolation
                logger.warning(f"编码错误：{file}")

    all_rst = glob.glob(f'{api_dir}/**/*.{origin}', recursive=True)

    for i in all_rst:
        if os.path.dirname(i).endswith("api_python"):
            continue
        name = os.path.basename(i)
        if name in tar:
            os.rename(i, i.replace(f'.{origin}', f'.{replace}'))
