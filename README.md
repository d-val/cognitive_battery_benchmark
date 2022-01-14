 
# Human Cognitive Battery Benchmark

This repository contains the simulated implementation (using [AI2-THOR](https://github.com/allenai/ai2thor) and Unity 3D) a series of cognitive science experiments that [are routinely run on monkeys, crows, humans, etc](https://dx.plos.org/10.1371/journal.pone.0032024 ).


## 💻 Installation


### Clone this repo
```python
git clone https://github.com/d-val/cognitive_battery_benchmark
```

- Download our customized build from the following link [here](https://www.dropbox.com/s/dd0goyeihrwaxe6/thor-OSXIntel64-local.zip?dl=0)
- Unzip the downloaded thor-OSXIntel64-local.zip file in the `cognitive_battery_benchmark/experiments/utils` folder

### Python 3.7 or 3.8 set-up:
#### With pip:

```bash
pip install -r setup/requirements.txt
```

#### With conda:

```bash
conda env create -f setup/environment.yml
conda activate cognitive-battery-benchmark
```

## Running
#### Minimal Example to test if AI2-THOR installation:

Most of our simulated environments are built on top of the excellent [AI2-THOR](https://github.com/allenai/ai2thor) interactable framework for embodied AI agents. 
After running `pip` or `conda` install earlier, it should have installed AI2-THOR. you can verify that everything is working correctly by running the following minimal example:

```python
python ai2-thor-minimal-example.py
```

This will download some AI2-THOR Unity3D libraries which can take a while as they are big (~0.5 GB)

```angular2html
ai2thor/controller.py:1132: UserWarning: Build for the most recent commit: 47bafe1ca0e8012d29befc11c2639584f8f10d52 is not available.  Using commit build 5c1b4d6c3121d17161935a36baaf0b8ac00378e7
  warnings.warn(thor-OSXIntel64-5c1b4d6c3121d17161935a36baaf0b8ac00378e7.zip: [|||||||||||||||||||||||||||||||||||||||                                                        42%   1.7 MiB/s]  of 521.MB
```
After this downloads, you will see a Unity simulator window open up:

<img src="static/unity.png" width="200">

Followed by this terminal output:

```angular2html
success! <ai2thor.server.Event at 0x7fadd0b87250
    .metadata["lastAction"] = RotateRight
    .metadata["lastActionSuccess"] = True
    .metadata["errorMessage"] = "
    .metadata["actionReturn"] = None
> dict_keys(['objects', 'isSceneAtRest', 'agent', 'heldObjectPose', 'arm', 'fov', 'cameraPosition', 'cameraOrthSize', 'thirdPartyCameras', 'collided', 'collidedObjects', 'inventoryObjects', 'sceneName', 'lastAction', 'errorMessage', 'errorCode', 'lastActionSuccess', 'screenWidth', 'screenHeight', 'agentId', 'colors', 'colorBounds', 'flatSurfacesOnGrid', 'distances', 'normals', 'isOpenableGrid', 'segmentedObjectIds', 'objectIdsInBox', 'actionIntReturn', 'actionFloatReturn', 'actionStringsReturn', 'actionFloatsReturn', 'actionVector3sReturn', 'visibleRange', 'currentTime', 'sceneBounds', 'updateCount', 'fixedUpdateCount', 'actionReturn'])
```



#### Running Human Cognitive Battery Experiment `SimpleSwap`
To run the experiment:

```
cd experiments/dataset_generation
python simple_swap_example.py
```
If success, a window will pop up, and the experiment will run in the terminal.

<img src="static/simpleswap.png" width="200">


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
  
## To run:
- 


## Issues & Debugging
- When running on OS X, you might get a prompt that `thor-OSXIntel64-local` is not verified. Follow the following [steps](https://support.apple.com/guide/mac-help/open-a-mac-app-from-an-unidentified-developer-mh40616/mac) for allowing running of the file. 