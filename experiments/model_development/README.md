# Baseline Model Testing
This directory contains scripts for training and evaluating baseline models consisting of a CNN and LSTM. 

## Dependencies
Required Python libraries are available in `requirements.txt`. You can create a virtual environment with the libraries installed.
```
conda create -y -n cog-battery-baseline pip
conda activate cog-battery-baseline
pip install -r requirements.txt
```

## Running Code
To run a training job, ensure that you have completed dataset generation and saved the output to `data/`; then, navigate to the Video Swin Transformer subdirectory and execute the training shell script.
```
cd utils/models/Video_Swin_Transformer
sh train.sh
```
