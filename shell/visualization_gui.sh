#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script="$CURRENT_DIR/../src/visualization_gui.py"
visual_config="$CURRENT_DIR/../configs/visualization.yaml"
hloc_config="$CURRENT_DIR/../configs/hloc.yaml"

conda activate UNav

python $script -v $visual_config -l $hloc_config
