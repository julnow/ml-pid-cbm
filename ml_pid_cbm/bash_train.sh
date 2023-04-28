#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cbm22

CONFIG="config.json"

#positive and negative particles
for i in 0 2 4 
do
    python -u ../../train_model.py -c $CONFIG -p $i $((i+2)) --saveplots --nworkers 10 | tee train_from_$i.txt
done
for i in 6 9 
do
    python -u ../../train_model.py -c $CONFIG -p $i $((i+3)) --saveplots --nworkers 10 | tee train_from_$i.txt 
done
