#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cbm22

CONFIG="config.json"

#positive and negative particles
for i in 0 3 6 9 
do
    python train_model.py -c $CONFIG -p $i $((i+3)) --saveplots --hyperparams | tee train_from_$i.txt
    # python train_model.py -c $CONFIG -p $i $((i+3)) --saveplots --antiparticles 
done
