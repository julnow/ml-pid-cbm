#!/bin/bash
LOGDIR=/lustre/cbm/users/$USER/pid/log
mkdir -p $LOGDIR
mkdir -p $LOGDIR/out
mkdir -p $LOGDIR/error

sbatch --job-name=all \
        -t 06:00:00 \
        --partition main \
        --output=$LOGDIR/out/%j.out.log \
        --error=$LOGDIR/error/%j.err.log \
        --array 1-4 \
        -- $PWD/slurm_ml_pid.sh
