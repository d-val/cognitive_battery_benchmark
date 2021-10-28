import os
from PIL import Image
from ai2thor.controller import Controller
from collections import namedtuple
import random
import yaml
import sys

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Experiment(Controller):
    def __init__(self, controller_args):
        super().__init__(
            **{
                **{  # local build
                    "local_executable_path": f"{BASE_DIR}/utils/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
                    "agentMode": "default",
                    "scene": "FloorPlan1",
                    # step sizes
                    "gridSize": 0.25,
                    "snapToGrid": False,
                    "rotateStepDegrees": 90,
                    # image modalities
                    "renderDepthImage": False,
                    "renderInstanceSegmentation": False,
                    # camera properties
                    "width": 300,
                    "height": 300,
                    "makeAgentsVisible": False,
                },
                **controller_args,
            }
        )
        self.frame_list = []
        self.saved_frames = []
        self.third_party_camera_frames = []

    def save_frames_to_folder(self, SAVE_DIR, first_person=True, save_stats=True):

        if not os.path.isdir(f"{SAVE_DIR}"):
            os.makedirs(f"{SAVE_DIR}")
        if save_stats:
            with open(
                f"{SAVE_DIR}/experiment_stats.yaml",
                "w",
            ) as yaml_file:
                yaml.dump(self.stats, yaml_file, default_flow_style=False)

        fov_frames = self.frame_list if first_person else self.third_party_camera_frames
        # height, width, channels = fov_frames[0].shape
        for i, frame in enumerate(fov_frames):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/frame_{i}.jpeg")

    def run(self):
        raise NotImplementedError
