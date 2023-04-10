#!/bin/bash

conda init bash
conda activate cbm22

CONFIG="config.json"

#validation of single models
for dir in model_*
    do
        if [[ -d "$dir" ]]; then
            readymodels+="$dir "
            # echo $dir
            #python validate_model.py -c $CONFIG -m $dir -p .9 .96 .94
        fi
done
#manual proba values
python validate_model.py -c $CONFIG -m model_0.0_3.0_positive -p .92 .95 .9
python validate_model.py -c $CONFIG -m model_3.0_6.0_positive -p .9 .9 .85
python validate_model.py -c $CONFIG -m model_6.0_9.0_positive -p .6 .8 .55
python validate_model.py -c $CONFIG -m model_9.0_12.0_positive -p .6 .6 .6
#validation of all models
python validate_multiple_models.py -c $CONFIG -m $readymodels
