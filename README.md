 
# Cognitive Battery Benchmark

## 💻 Installation

#### With pip:

```bash
pip install -r setup/requirements.txt
```

#### With conda:

```bash
conda env create -f environment.yml
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
By installing ai2thor package, it should automatically download all the required packages. You can check them in 'requirements.txt'
## Requirements
Python 3.7 or 3.8

## Clone this repo
```python
git clone https://github.com/d-val/cognitive_battery_benchmark/tree/neil_test
```

## Running 3 Pots 1 Egg

Download our customized build from the following link [here](https://www.dropbox.com/s/jf69rhi08a7ve0r/thor-OSXIntel64-local.zip?dl=0)

Unzip thor-OSXIntel64-local.zip file at the 'cognitive_battery_benchmark' folder

Run `aithor_test.py`
```python
python3 aithor_test.py
```

If success, a window with size 2000x2000 will pop up. The egg will be put randomly into one pot.

## Module Structure

The structure of the 'cognitive_battery_benchmark' folder is as follows:

- aithor_test.py: core movements/instruction for the AITHOR agent to perform 3 pots 1 egg
- util.py: helper functions for agent movements
