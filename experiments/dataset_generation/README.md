# Dataset Generation for AI2-THOR

## Run Datageneration Jobs:

There are three main files that are used to generate a dataset:
* Experiment Code
* Experiemnt Config
* Render Config

### Experiment Code
The experiment code is the python file that contains the logic for generating the dataset.
Here is an example of the experiment code to use `config/linux_renderer.yaml` as the renderer file
and `config/Shape_config.yaml` as the experiment config file.

``` python
from utils.experiment_job import ExperimentJob
if __name__ == "__main__":
    expt = ExperimentJob(renderer_file="config/linux_renderer.yaml", 
                        experiment_files=["config/Shape_config.yaml"],
                        test_init=False,
                        test_run=False)
    expt.run(folder_name="Shape", run_name="Run0", seed_pattern='iterative')
```

The main arguments for `ExperimentJob` initialization are:
* `renderer_file`: The path to the renderer config file.
  * The render details are the lowest level of the config hierarchy, and can be overwritten by specific render details
    in the experiment config files.
* `experiment_files`: A list of paths to the experiment config files.
There are two main ways of structuring a multiple experiment run:
  * One experiment config file with multiple experiments
  * Multiple experiment config files with `>=1` experiments each, and listing them in the `experiment_files` argument.
* `test_init`: A boolean flag to test the initialization of the experiment code. This is useful for debugging.
* `test_run`: A boolean flag to test the run of the experiment code. This is useful for debugging.

The main arguments for `ExperimentJob.run` are:
* `folder_name`: The name of the main folder to save the dataset in, corresponding to the main folder for runs.
* `run_name`: The name of the subfolder to save the dataset in, corresponding to this specific run.
* `seed_pattern`: The pattern to use for the random seed. The options are:
  * `iterative`: The seed will be the iteration number.
  * `random`: The seed will be a random number.
  * `("fixed", seed)`: The seed will be a fixed number, specified by the `seed` argument.


### Experiment Config
The experiment config file is a yaml file that contains the details for the experiment. 
Here is an example of the experiment config file for generating a dataset of shapes.

``` yaml
SimpleSwap:
  iterations: 250
  controllerArgs:
    renderDepthImage: On
    renderInstanceSegmentation: On
  init:
    fov: 90
  testing_parameters:
    swaps: [1, 2, 3, 4, 5, 6, 7, 8]
  run:
    moveup_magnitude: 0.45
    move_recep_ahead_mag: 0.45
    receptacleType: null
    receptacleTypes: ["Pot", "Mug"] # ["Pot", "Mug", "Cup"]
    reward_pot: null
    rewardType: null
    rewardTypes: ["Egg", "Potato", "Tomato", "Apple"]
    pots_to_swap: null
    reward_position: null
    num_receptacles: 6
```
The structure of the experiment config file is as follows:
* The name of the experiment class (e.g. `SimpleSwap`), the code will use this name to find the experiment class, so it should match.
  * `iterations`: The number of iterations to run for this experiment, each iteration will generate a new scene.
  * `controllerArgs`: The arguments to pass to the controller. These arguments will be passed to the `__init__` function of the controller.
  * `init`: The initialization parameters for the experiment. These parameters will be passed to the `__init__` function of the experiment class.
  * `testing_parameters`: The testing parameters for the experiment. These parameters will be passed to the `run` function of the experiment class.
  These parameters are variations to the parameters in `run` that will be used to generate multiple different 
  versions of the experiment. Each parameter in `testing_parameters` should be a list of values to test, and can be 
  any parameter found in the `run` function of the experiment class. Each of these parameters will result in a folder with
  `iterations` number of scenes.
  * `run`: The run parameters for the experiment. These parameters will be passed to the `run` function of the experiment class.
  These are different from the `testing_parameters` in that they are the parameters that they will hold constant for the entire run,
    and will not be varied, unlike `testing parameters`.

### Render Config
The render config file is a yaml file that contains the details for the renderer.
Here is an example of the render config file for generating a dataset of shapes.

``` yaml
# local build
local_executable_path: "utils/test.app/Contents/MacOS/AI2-THOR"
agentMode: "default"
scene: "FloorPlan1"
platform: CloudRendering

# step sizes
gridSize: 0.25
snapToGrid: Off
rotateStepDegrees: 90

# image modalities
renderDepthImage: On
renderInstanceSegmentation: On

# camera properties
width: 300
height: 300
makeAgentsVisible: Off
```

These properties are the same as the properties in the `controllerArgs` in the experiment config file, and will be passed to the controller.
They are defined by ai2thor, and can be found in the [ai2thor documentation](https://ai2thor.allenai.org/robothor/documentation/#initialization).
The main option that will probably be changed is the `local_executable_path`, which is the path to the ai2thor executable.
This path will be different depending on the OS and the build of ai2thor that is being used.

## Run on Server

### Setup screen

Run the `linux_setup.py` script in the `setup` folder in the main directory
to setup the screen session in a background process. Once you've confirmed that the screen session
is running, you can run the experiment (usually in the background to avoid interupts
from the screen session).

### Run experiment

Following the experiment setup instructions above, to
setup the renderer config, experiment config, and experiment code,
run the experiment script in the background.


# Unity Generation

### Download
* Download the custom ai2thor Unity folder at https://www.dropbox.com/s/n54s85ynpo6dpr6/unity.zip?dl=0 and unzip it.

* Download [UnityHub](https://unity.com/unity-hub) and install Unitity version 2020.3.25f1

### Modify + Build
* Open UnityHub and click on the "Add" button to add the unity folder you just downloaded.
* It will take a bit to process.
* Once open, make the modifications that you'd like to make to the scene.
* Then, when ready, click on the "Build" button and select the "Build" folder.
* Select the OS you are building for and click "Build".
* This will create a build of the scene in the "Build" folder, which can then be referenced directly in the config
data generation files.

### Download Prebuilt:
* Download the prebuilt Unity folder at https://www.dropbox.com/s/4j7q2qj4q2q2q2q/unity.zip?dl=0 and unzip it.

