#!/bin/sh
CURR_DIR=$( dirname "${BASH_SOURCE[0]}" )
CURR_DIR_S=`echo $CURR_DIR | awk -F '/to' '{print $1}'`
echo "shs into dir $CURR_DIR_S"
python $CURR_DIR/annotation_rulemessage.py $1 $2
