#!/bin/sh
CURR_DIR=$( dirname "${BASH_SOURCE[0]}" )
echo "sh into dir $CURR_DIR"
cd "$( dirname "${BASH_SOURCE[0]}" )"
python $CURR_DIR/annotation_rulemessage.py $1 $2
