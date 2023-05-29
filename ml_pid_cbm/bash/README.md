
## binary

In this folder, multiple useful bash scripts could be found.

### bash train

For example, in the [bash_training](../main/bash_train.sh) we can define:

```bash
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

CONFIG="config.json"
python -u ../../train_model.py -c $CONFIG -p 0 1.6 --saveplots --nworkers 8 --usevalidation  | tee train_bin_0.txt
python -u ../../train_model.py -c $CONFIG -p 1.6 2.3 --saveplots --nworkers 8 --usevalidation  | tee train_bin_1.txt
```

### bash validate

Later, in the [bash_validate](../main/bash_validate.sh):

```bash
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

CONFIG="config.json"

#validation of single models
for dir in model_*
    do
        if [[ -d "$dir" ]]; then
            readymodels+="$dir "
            python ../../validate_model.py -c $CONFIG -m $dir -n 8 -e .4 .95 40 -a 90
       fi
done
python ../../validate_multiple_models.py -c $CONFIG -m $readymodels --nworkers 4
```
which will validate all the models in the directory, and later merge their results.

### merge pdf

In the [merge_pdf](../main/merge_pdf.sh) a script for merging multiple output images into a single pdf with all the useful plots can be found.

### slurm
[slurm_ml_pid](../main/slurm_ml_pid.sh) is a script which could be run on the hpc.gsi.d batchfarm using [batchrun_train](../main/batchrun_train.sh) command.



