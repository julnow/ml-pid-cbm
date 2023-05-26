#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cbm22

CONFIG="config.json"

#validation of single models
for dir in model_*
    do
        if [[ -d "$dir" ]]; then
            readymodels+="$dir "
            # echo $dir
            # python ../../validate_model.py -c $CONFIG -m $dir -n 4 -e .4 .95 40 
       fi
done
#manual proba values
#python ../../validate_model.py -c $CONFIG -m model_0.0_2.0_positive -n 4 -p .4 .95 .95
#python ../../validate_model.py -c $CONFIG -m model_2.0_4.0_positive -n 4 -p .4 .4 .4
#python ../../validate_model.py -c $CONFIG -m model_4.0_6.0_positive -n 4 -p .4 .4 .4
#python ../../validate_model.py -c $CONFIG -m model_6.0_9.0_positive -n 4  -p .4 .95 .4
#python ../../validate_model.py -c $CONFIG -m model_9.0_12.0_positive -n 4 -p .4 .4 .4
#validation of all models
python ../../validate_multiple_models.py -c $CONFIG -m $readymodels --nworkers 4
