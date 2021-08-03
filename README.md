# nguyen_cognitive_battery
#How to create a benchmarking video sample

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

#### Requirements

| Component | Requirement |
| :-- | :-- |
| OS | Mac OS X 10.9+, Ubuntu 14.04+ |
| Graphics Card | DX9 (shader model 3.0) or DX11 with feature level 9.3 capabilities. |
| CPU | SSE2 instruction set support. |
| Python | Versions 3.5+ |
| Linux | X server with GLX module enabled |

## Running 3 Pots 1 Egg

Install Unity Editor version 2019.4.20 LTS for OSX (Linux Editor is currently in Beta) from [Unity Download Archive](https://unity3d.com/get-unity/download/archive).

Download our customized build from the following link [here](https://www.dropbox.com/l/scl/AAAlqUh3ySlx0FZGXI3kNBKqHzdwk9lCO7U)

Unzip thor-OSXIntel64-local.zip file

In `aithor_test.py`, change `BASE_DIR` to the directory that contains 'thor-OSXIntel64-local' application. For example:

```python
BASE_DIR = "/Users/dhaval/Documents/cognitive_battery_clean"
```

Run `aithor_test.py`
```python
python3 aithor_test.py
```