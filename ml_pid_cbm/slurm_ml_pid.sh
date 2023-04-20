#!/bin/bash

INDEX=${SLURM_ARRAY_TASK_ID}
WORK=/lustre/cbm/users/$USER
#path to ml-pid-cbm python package
mlpidpath=$WORK/ml-pid-cbm/ml_pid_cbm

#load conda
export PATH=/lustre/cbm/users/jnowak/miniconda3/bin/:$PATH
conda activate cbm23
#get into folder for training
CONFIG=$mlpidpath/test_config.json
cd $WORK/pid/
#run training
python $mlpidpath/train_model.py -c $CONFIG -p $(((INDEX-1)*3)) $((INDEX*3)) --saveplots | tee training_output_${INDEX}.txt
