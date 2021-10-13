 
# How to run battery test simulation

## 💻 Installation

#### With pip

```bash
pip install ai2thor
```

#### With conda

```bash
conda install -c conda-forge ai2thor
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
By installing ai2thor package, it should automatically download all the required packages. You can check them in 'requirements.txt'
## Requirements
Python 3.7 or 3.8

## Clone this repo
```python
git clone https://github.com/d-val/nguyen_cognitive_battery.git
```

## Running Simulation

Download our customized build from the following link [here](https://www.dropbox.com/s/jf69rhi08a7ve0r/thor-OSXIntel64-local.zip?dl=0)

Unzip thor-OSXIntel64-local.zip file at the 'nguyen_cognitive_battery' folder

To get a random example of a cognitive battery test, simply run the corresponding file. 
For example run
```python
python3 rotation.py
```

If success, a window with size 2000x2000 will pop up. A random reward object hiding under one of the three cups, all of which on a tray is shown. The tray then rotates 180 or 360 degree.

To customized, add flags. For instance in this example below would specify the reward object to be a potato. The tray would be under the middle cup; the tray rotates 180 degree as in subtask 1 of rotation test.

```python
python3 rotation.py --reward 0 --case 1
```

To learn more about available flags, run `-h` or `--help` flag
```python
python3 rotation.py -h
```
or
```python
python3 rotation.py --help
```

## Saving Images

To save images of a simulation, uncomment the last line
```python
    def save_frames_to_file(self):
            from PIL import Image
    
            image_folder = './'
            print('num frames', len(self.frame_list))
            height, width, channels = self.frame_list[0].shape
    
            for i, frame in enumerate(tqdm(self.frame_list)):
                img = Image.fromarray(frame)
                ###REPLACE THE FOLLOWING LINE WITH THE DIRECTORY WHERE THE IMAGES TO BE SAVED
                img.save("rotation_agent_view/{}.jpeg".format(i))
            
            print('num frames', len(self.third_party_camera_frames))
            height, width, channels = self.third_party_camera_frames[0].shape
    
            for i, frame in enumerate(tqdm(self.third_party_camera_frames)):
                img = Image.fromarray(frame)
                ###REPLACE THE FOLLOWING LINE WITH THE DIRECTORY WHERE THE IMAGES TO BE SAVED
                img.save("rotation_monkey_view/{}.jpeg".format(i))

vid = VideoBenchmark()
### UNCOMMENT THE FOLLOWING LINE TO SAVE IMAGES
# vid.save_frames_to_file()
```
## Module Structure

The structure of the 'nguyen_cognitive_battery' folder is as follows:

- aithor_test.py: core movements/instruction for the AITHOR agent to perform 3 pots 1 egg
- util.py: helper functions for agent movements
