# About

This implementation of [TimeSformer](https://arxiv.org/pdf/2102.05095.pdf) is derived from [the official code repo](https://github.com/facebookresearch/TimeSformer).

# Setup for running on GCP

## Local setup

### Download pretrained model from [official TimeSformer repo](https://github.com/facebookresearch/TimeSformer)

Create a new directory, `./experiments/model_development/utils/models/TimeSformer/pretrained/` (relative to root of repo), and save the model here.

The model used in this project is [TimeSformer-L](https://www.dropbox.com/s/r1iuxahif3sgimo/TimeSformer_divST_96x4_224_K400.pyth).

### Transfer files

Run the following commands locally:

```
cd experiments/model_development/ # Navigate from root of repo; skip if already here

gcloud compute scp --recurse data/ your_gcp_username:/home/your_gcp_username/cognitive_battery_benchmark/experiments/model_development
gcloud compute scp --recurse utils/models/TimeSformer/pretrained/ your_gcp_username:/home/your_gcp_username/cognitive_battery_benchmark/experiments/model_development/utils/models/TimeSformer
```

## GCP setup

### Connect to GCP

First, create and start a VM instance [on GCP](https://console.cloud.google.com/compute/instances?project=bridge-urops).

Then, run the following command locally to SSH to GCP:

```
gcloud compute ssh "your-gcp-username" --project "bridge-urops" --zone "your-zone" # SSH to GCP; zone switch not required if default config was set up during GCP CLI install
```

### Set up virtual environment

```
sudo apt update # may not be necessary if attempting below installs does not produce an error
sudo apt install python3-pip
sudo apt install python3.8-venv

python3 -m pip install --user --upgrade pip # may not be necessary if attempting below installs does not produce an error
python3 -m pip install --user virtualenv

python3 -m venv env
```

Activate the environment; note that this will need to be run every time, not just during setup!

```
source env/bin/activate
```

### Install requirements

```
cd cognitive_battery_benchmark/ # navigate to root of repo
pip3 install -r requirements.txt
```

### Follow [original TimeSformer setup instructions](https://github.com/facebookresearch/TimeSformer)

Install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`

Lastly, build the TimeSformer codebase by running:
```
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
python setup.py build develop
```

### Run training script

```
cd ./experiments/model_development
python3 train.py
```
