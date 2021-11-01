 
# Cognitive Battery Benchmark

## 💻 Installation

#### With pip:

```bash
pip install -r setup/requirements.txt
```

#### With conda:

```bash
conda env create -f setup/environment.yml
conda activate cognitive-battery-benchmark
```

#### With Docker

[🐳 AI2-THOR Docker](https://github.com/allenai/ai2thor-docker) can be used, which adds the configuration for running a X server to be used by Unity 3D to render scenes.

#### Minimal Example

Once you've installed AI2-THOR, you can verify that everything is working correctly by running the following minimal example:

```python
from ai2thor.controller import Controller
controller = Controller(scene="FloorPlan10")
event = controller.step(action="RotateRight")
metadata = event.metadata
print(event, event.metadata.keys())
```
By installing the ai2thor package, it should automatically download all the required packages. You can check them in `requirements.txt`.

## Clone this repo
```python
git clone https://github.com/d-val/cognitive_battery_benchmark
```
## Requirements
Python 3.7 or 3.8

- Download our customized build from the following link [here](https://www.dropbox.com/s/jf69rhi08a7ve0r/thor-OSXIntel64-local.zip?dl=0)
- Unzip the downloaded thor-OSXIntel64-local.zip file in the `cognitive_battery_benchmark/experiments/utils` folder

OR

From inside the root `cognitive_battery_benchmark` folder:
```commandline
wget https://www.dropbox.com/s/jf69rhi08a7ve0r/thor-OSXIntel64-local.zip
unzip thor-OSXIntel64-local.zip -d experiments/utils
rm thor-OSXIntel64-local.zip
```

## Running Example Experiment [`SimpleSwap`]

Run `cd experiments`:
```python

from simple_swap import SimpleSwap 

SimpleSwapExperiment = SimpleSwap()
SimpleSwapExperiment.run()
SimpleSwapExperiment.save_frames_to_folder('output')
```

If success, a window with size `2000x2000` will pop up, and the experiment will run in the terminal.
## Saving Images

To save images of a simulation, uncomment the last line
```python
SimpleSwapExperiment.save_to_folder()
### UNCOMMENT THE FOLLOWING LINE TO SAVE IMAGES
# vid.save_frames_to_file()

```
## Module Structure

The structure of the `cognitive_battery_benchmark` folder is as follows:

- **experiments**: files necessary to run experiments, based on reference paper
  - `addition_numbers.py`: addition numbers test class
  - `experiment.py`: base experiment class
  - [**not working**] `object_permanence.py`: object permanence test class 
  - [**not working**] `relative_numbers.py`: relative numbers test class
  - `rotation.py`: rotation swap test class
  - `simple_swap.py`: simple swap test class
  - **utils**: helper files for experiments
    - `util.py`: helper functions
    - `video_controller.py`: basic implementation for a video controller for AI2THOR  
  - **frames**: folder to store frames of experiments
- **setup**: helper files for setting up the module
  - `environment.yml`: setup conda installation
  - `requirements.txt`: setup pip installation
  

## Issues & Debugging
- When running on OS X, you might get a prompt that `thor-OSXIntel64-local` is not verified. Follow the following [steps](https://support.apple.com/guide/mac-help/open-a-mac-app-from-an-unidentified-developer-mh40616/mac) for allowing running of the file. 