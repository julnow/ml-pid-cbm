#!/bin/bash

conda init bash
conda activate cbm22

CONFIG="config.json"

#validation of single models
for dir in model_*
    do
        if [[ -d "$dir" ]]; then
            readymodels+="$dir "
            echo $dir
            python validate_model.py -c $CONFIG -m $dir -p .9 .95 .9 
        fi
done
#validation of all models
python validate_multiple_models.py -c $CONFIG -m $readymodels
