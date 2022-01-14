# -*- coding: utf-8 -*-
import os
import random

import numpy as np

# unity directory
from experiment import Experiment
from utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Shape(Experiment):
    def __init__(
        self,
        controller_args={
            "local_executable_path": "utils/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
            "agentMode": "default",
            "scene": "FloorPlan1",
            "gridSize": 0.25,
            "snapToGrid": False,
            "rotateStepDegrees": 90,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": 1000,
            "height": 1000,
            "makeAgentsVisible": False,
        },
        fov=[90, 120],
        visibilityDistance=5,
        seed=0,
    ):

        random.seed(seed)
        np.random.seed(seed)
        self.stats = {
            "visibility_distance": visibilityDistance
            if type(visibilityDistance) != list
            else random.randint(*visibilityDistance),
            "fov": fov if type(fov) != list else random.randint(*fov),
        }
        super().__init__(
            {
                **{
                    # local build
                    "visibilityDistance": self.stats["visibility_distance"],
                    # camera properties
                    "fieldOfView": self.stats["fov"],
                },
                **controller_args,
            }
        )

        self.step(
            action="AddThirdPartyCamera",
            position=dict(x=1.5, y=1.5, z=0),
            rotation=dict(x=0, y=270, z=0),
            fieldOfView=90,
        )

        # Randomize Materials in the scene
        self.step(action="RandomizeMaterials")

        # Randomize Lighting in the scene
        self.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False,
        )

    def run(
        self, rewardTypes=["Potato", "Tomato", "Apple"], rewardType=None, coveringTypes=["Plate", "Bowl"], coveringType=None, max_reward=6
    ):
        # TODO: add ability to specify number of items in each plate
        self.rewardType, self.coveringType = (
            random.sample(rewardTypes, 1)[0] if rewardType is None else rewardType,
            random.sample(coveringTypes, 1)[0] if coveringType is None else coveringType,
        )

        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        afterPoses = []

        # Initialize Object by specifying each object location, receptacle and reward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        for obj in self.last_event.metadata["objects"]:
            # current Pose of the object
            initialPose = {
                "objectName": obj["name"],
                "position": obj["position"],
                "rotation": obj["rotation"],
            }

            afterPose = {
                "objectName": obj["name"],
                "position": obj["position"],
                "rotation": obj["rotation"],
            }

            # Set the Plates location (pre-determined)
            if obj["objectType"] == "Box":
                cardboard1 = obj["objectId"]
                # right Cardboard1 (z > 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.15, "y": 1.105, "z": 0.45},
                    }
                )

                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.15, "y": 1.105, "z": -0.45},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.15, "y": 1.205, "z": 0.45},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.15, "y": 1.205, "z": -0.45},
                    }
                )

            elif obj["name"] == "Occluder":
                # right CausualityOccluder1
                occluder1 = obj["objectId"]
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0.4, "y": 1.3587, "z": 0.4},
                    }
                )

                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0.4, "y": 1.3587, "z": -0.28},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.55, "y": 1.3587, "z": 0.4},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.55, "y": 1.3587, "z": -0.28},
                    }
                )

            elif obj["objectType"] == self.rewardType:
                initialPoses.append(initialPose)
                self.out = random.choice([1, -1])
                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.172, "y": 1.15, "z": self.out * 0.45},
                    }
                )
            else:
                initialPoses.append(initialPose)
                afterPoses.append(afterPose)

        # set initial Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        for obj in self.last_event.metadata["objects"]:
            print(obj["objectType"])
            if obj["name"] == "Occluder":
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.2), (-0.95, 0, 0), (0, 0, -0.2)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )
            elif "Occluder" in obj["name"]:
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.2), (-0.95, 0, 0), (0, 0, -0.2)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )

        self.step(
            action="SetObjectPoses", objectPoses=afterPoses, placeStationary=False
        )

        for obj in self.last_event.metadata["objects"]:
            if obj["name"] == "Occluder":
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.1), (0, -1, 0)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )
            elif "Occluder" in obj["name"]:
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.1), (0, 1, 0)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )
        # dummy moves for debugging
        # self.step("MoveBack", moveMagnitude=0)
        # self.step("MoveAhead", moveMagnitude=0)
        if self.last_event.metadata["errorMessage"]:
            print(f'ERROR1:{self.last_event.metadata["errorMessage"]}')
        # count rewards to get output
        self.label = self.out


x = Shape()
x.run()
# x.save_frames_to_folder("shape")
