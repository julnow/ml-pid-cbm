#!/bin/bash

conda init bash
conda activate cbm22

CONFIG="config.json"

#validation of single models
for f in ./model_*
do
    python validate_model.py -c $CONFIG -m $f -p .9 .95 .9 
done
#validation of all models
models=$(find . -maxdepth 1 -type d -name 'model_*' -exec basename {} \;)
python validate_multiple_models.py -c $CONFIG -m $models
