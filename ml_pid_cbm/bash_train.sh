#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cbm22

CONFIG="config.json"

#positive and negative particles
for i in 0 2 4 6 
do
    python -u ../../train_model.py -c $CONFIG -p $i $((i+2)) --saveplots --nworkers 16 --usevalidation --hyperparams | tee train_from_$i.txt
done
python -u ../../train_model.py -c $CONFIG -p 8 12 --saveplots --nworkers 16 --usevalidation --hyperparams | tee train_from_8.txt
