# Baseline Model Testing
This directory contains scripts for training and evaluating baseline models consisting of a CNN and LSTM. 

## Dependencies
Required Python libraries are available in `requirements.txt`. You can create a virtual environment called `cog-battery-baseline` (or any other name you choose) with the libraries installed.

### Option 1: set up virtual environment with conda
```
conda create -y -n cog-battery-baseline pip
conda activate cog-battery-baseline
pip install -r requirements.txt
```

### Option 2: set up virtual environment with vemv
```
python3 -m venv cog-battery-baseline
source cog-battery-baseline/bin/activate
pip3 install -r requirements.txt
```

See [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments) for more information on venv.

## Running Code
To run a sample training job, you need to
1. Specify the model architecture and training parameters in `config/config.yaml`. A description of the config can be found below.
2. Add the data on which you'd like to run the training job.
    A toy example of a data directory is available [here](https://www.dropbox.com/s/rh8i0gblyljbz0j/GravityBias.zip?dl=0).

    If you are unable to access the toy dataset, you can generate data by running the following within your activated virtual environment:
    ```
    cd ../dataset_generation/
    pip3 install -r requirements.txt
    python3 run_all_experiments.py
    ```
    Ensure that the number of iterations set in `dataset_generation/config/AllExperiments_config.yaml` provides sufficient data for your training job.
3. Copy the data from the previous step into this directory in the following format:
    ```
      data/
           0/
              machine_readable/
                    iteration_data.pickle
           1/
              machine_readable/
                    iteration_data.pickle
            ⋮
    ```
    Alternatively, you may specify a path to a folder following the previous format in `config/config.yaml`. For instance, you might set `data_path: "../dataset_generation/output/20*/Shape/"` rather than `data_path: "data/"` under `data_loader`.
3. Run `train.py`.

This will run a training job with the specified name and save the resulting log(s) and model(s) in the `output` directory.

## Config Parameters
* `job_name`: a name used to identify the output of the current job.
* `expt_name`: the name of the experiment for which the data belongs. For now, it must be `"gravity"`, `swap`, or `shape`.
* `model`: contains a description of the model. Should specify:
  * `cnn_architecture`: name of CNN architecture. For now, supports `resnet18`, `resnet34`, and `alexnet`. 
  * `lstm_num_layers`: number of layers in the LSTM block.
  * `lstm_hidden_size`: the hidden size of the LSTM block.
  * `num_classes`: the number of output classes.
* `data_loader`: contains the config for the data loading mechanism.
  * `data_path`: the path where data is loaded. Typically, it should be `"data/"`.
  * `batch_size`: how many videos to load per training pass.
  * `train_split`: the portion of data to use for training (e.g. `0.8` uses %80 of data for training and %20 for testing).
* `train_params`: contains some parameters of the training job.
  * `epochs`: how many transits through the training data for this training job.
  * `lr`: the gradient step size at each iteration.

An example config file is available in `config/config.yaml`.
