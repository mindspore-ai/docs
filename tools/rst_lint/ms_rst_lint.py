"""Run rst_lint."""
import subprocess
import sys
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s')
    process = subprocess.Popen(['python', 'run.py', *sys.argv[1:]],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               encoding="utf-8")
    stdout, _ = process.communicate()

    for i in stdout.split('\n'):
        if "Duplicate explicit target name:" in i:
            continue
        logging.warning(i)
