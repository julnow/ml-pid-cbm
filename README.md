
# ml-pid-cbm


 [![codecov](https://codecov.io/gh/julnow/ml-pid-cbm/branch/main/graph/badge.svg)](https://codecov.io/gh/julnow/ml-pid-cbm)


Python package for training ML model for particle Identification (Msc thesis) in the CBM experiment.

This package is based on the [hipe4ml](https://hipe4ml.github.io) package.



## Installation

To run this project, you need to set up a Conda environment with the required packages. Follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/julnow/ml-pid-cbm.git
   cd ml-pid-cbm
   ```
2. Install necessary packages described in the [enivornment.yml](../main/environment.yml) file to your conda environment, for example:
    ```bash
    conda env update --file environment.yml --name enivornment_name
    ```
3. As this package is based on the hipe4ml, Mac OS users also are required to install the OpenMP library:
    ```bash
    brew install libomp
    ```

## Run Locally

This package constsits of three main modules.

### 0. config.json
However, you should first  fill all the necessary fields in the [config.json](../main/ml_pid_cbm/config.json)
The root trees can be e.g., created using [ml-tree-plainer package](https://github.com/julnow/ml-tree-plainer).

```json
{
    "file_names": {
        "training": "/path/to/traing/dataset.root",
        "test": "/path/to/test_validation/dataset.root"
    },
    "var_names": {
        "momentum": "name_of_the_momentum_variable_in_tree"
    },
    "features_for_train": [ "mass2", "dE/dx"],
    "vars_to_draw": ["variable_a","xgb_preds"],
    "cuts": {
        "momentum": {"lower": -12.0, "upper": 12.0},
    },
    "hyper_params": {
        "values": {"n_estimators": 670},(...)
        "ranges": {
            "n_estimators": [300, 1200],(...)
        }
    }
}
```
If the hyper_params are given explicitly, the model can use them; providing the ranges is necessary for the optimization of hyperparams wiht optuna.
### 1.  train_model 
Module for training the XGBoost model.

It should be run with options:

```bash
usage: ML_PID_CBM TrainModel [-h] --config CONFIG --momentum MOMENTUM MOMENTUM
                             [--antiparticles] [--hyperparams] [--gpu]
                             [--nworkers NWORKERS]
                             [--printplots | --saveplots] [--usevalidation]
```
where:
* `--config` should be the location of the config file
* `--momentum` describe lower and upper momentum cut
* `--antiparticles` flag sets used only  negative charge, otherwise positive
* `--gpu` turns on GPU-usage for training
* `--nworkers` sets number of threads available for the _ThreadPoolExecutor_
* `--printplots` shows the plots interactively, while the `--saveplots` saves them in png and pdf format
* `--usevalidation` uses validation dataset for creating the model output graphs, useful to check during the training if the model performs similarily on the training validation (e.g., created using DCM simulation model) and validation (e.g., creating using URQMD)

### 2.  validate_model
Module for validating a trained XGBoost model.

It should be run with options:

```bash
usage: ML_PID_CBM ValidateModel [-h] --config CONFIG --modelname MODELNAME
                                (--probabilitycuts PROBABILITYCUTS PROBABILITYCUTS PROBABILITYCUTS | --evaluateproba EVALUATEPROBA EVALUATEPROBA EVALUATEPROBA)
                                [--nworkers NWORKERS]
                                [--interactive | --automatic AUTOMATIC]
```
where:
* `--config` should be the location of the config file
* `--modelname` is the name of the folder created during the trainig step containg the model (which will have the same name).
* `--nworkers` sets number of threads available for the _ThreadPoolExecutor_
* Probabilitycuts:
  * `--probabilitycuts` can be set manually, for respectively PROTONS, KAONS, PIONS in the current implementation, e.g., .9 .8 .9
  * `--evaluateproba` will check probability cuts for each particle from LOWER_VALUE to UPPER_VALUE using N_STEPS, e.g., .35 .98 40
* If probabilitycuts where set using `--evaluateproba`, user have to options:
    * Select them interactively if `--interactive` provided
    * Apply automatic selection, aiming for MINIMAL_PURITY %, e.g., 90

#### The automatic probabilitycut selection alghoritm:
  1. Chooses the probability cut with the highest efficiency, if the purity is higher than MINIMAL_PURITY
  2. If there is no cut with purity> MINIMAL_PURITY, it will choose the one with the highest purity. 

### 3.  validate_multiple_models
Module for merging the results from multiple models into single output and histograms.

It should be run with options:
```bash
usage: ML_PID_CBM ValidateMultipleModels [-h] --modelnames MODELNAMES
                                         [MODELNAMES ...] --config CONFIG
                                         [--nworkers NWORKERS]
```
where:
* `--config` should be the location of the config file
* `--nworkers` sets number of threads available for the _ThreadPoolExecutor_
* `--modelnames` should be a list of all validated models whose results should be merged, e.g., `-m modelA modelB modelC`


### 4. Bash files
For better automation, a bash file can be created for all the steps.

For example, in the [bash_training](../main/ml_pid_cbm/bash/bash_train.sh) we can define:

```bash
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

CONFIG="config.json"
python -u ../../train_model.py -c $CONFIG -p 0 1.6 --saveplots --nworkers 8 --usevalidation  | tee train_bin_0.txt
python -u ../../train_model.py -c $CONFIG -p 1.6 2.3 --saveplots --nworkers 8 --usevalidation  | tee train_bin_1.txt

```
Later, in the [bash_validate](../main/ml_pid_cbm/bash/bash_validate.sh):

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


## Documentation

Documentation available [here](https://julnow.github.io/ml-pid-cbm/ml_pid_cbm.html)
