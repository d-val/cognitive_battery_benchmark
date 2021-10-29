# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
from tqdm import tqdm
import math

# unity directory
from experiment import Experiment
from utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Shape(Experiment):
    def __init__(self, fov=None, seed=0):

        # random.seed(seed)

        super().__init__(
            {
                # local build
                "local_executable_path": f"{BASE_DIR}/utils/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
                "agentMode": "default",
                "visibilityDistance": 5,
                "scene": "FloorPlan1",
                # step sizes
                "gridSize": 0.25,
                "snapToGrid": False,
                "rotateStepDegrees": 90,
                # image modalities
                "renderDepthImage": False,
                "renderInstanceSegmentation": False,
                # # camera properties
                "width": 2000,
                "height": 2000,
                "fieldOfView": random.randint(90, 120) if fov is None else fov,
                "makeAgentsVisible": True,
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
        self, rewardTypes=["Potato", "Tomato", "Apple"], rewardType=None, max_reward=6
    ):
        #TODO: add ability to specify number of items in each plate
        self.rewardType = random.sample(rewardTypes, 1)[0]

        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        afterPoses = []

        # Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
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
            if obj["name"] == "Cardboard1":
                
                cardboard1 = obj["objectId"]
                # right Cardboard1 (z > 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.172, "y": 1.105, "z": 0.45},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.172, "y": 1.205, "z": 0.45},
                    }
                )
            elif obj["name"] == "Cardboard2":
                # left Cardboard1 (z < 0)
                cardboard2 = obj["objectId"]
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.172, "y": 1.105, "z": -0.45},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.172, "y": 1.205, "z": -0.45},
                    }
                )

            elif obj["name"] == "CausualityOccluder1":
                # right CausualityOccluder1
                occluder1 = obj["objectId"]
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0.3, "y": 1.3587, "z": 0.45},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0.3, "y": 1.3587, "z": 0.45},
                    }
                )
            elif obj["name"] == "CausualityOccluder2":
                occluder2 = obj["objectId"]
                # left CausualityOccluder1
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0.3, "y": 1.3587, "z": -0.45},
                    }
                )

                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0.3, "y": 1.3587, "z": -0.45},
                    }
                )
            elif obj["objectType"] == "Potato":
                initialPoses.append(initialPose)
                i = random.choice([1,-1])
                afterPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.172, "y": 1.15, "z": i*0.45},
                    }
                )
            else:
                initialPoses.append(initialPose)
                afterPoses.append(afterPose)

        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        for obj in self.last_event.metadata["objects"]:
            if obj["name"] == "CausualityOccluder1" or  obj["name"] == "CausualityOccluder2":
                _, self.frame_list, self.third_party_camera_frames = move_object(
                            self,
                            obj["objectId"],
                            [(0, 0, 0.3),(0, 0, -0.3)],
                            self.frame_list,
                            self.third_party_camera_frames,
                        )
        self.step(
            action="SetObjectPoses", objectPoses=afterPoses, placeStationary=False
        )

        if self.last_event.metadata["errorMessage"]:
            print(self.last_event.metadata["errorMessage"])

        for obj in self.last_event.metadata["objects"]:
            if obj["name"] == "CausualityOccluder1":
                _, self.frame_list, self.third_party_camera_frames = move_object(
                            self,
                            obj["objectId"],
                            [(0, 0, 0.1),(0, -1, 0)],
                            self.frame_list,
                            self.third_party_camera_frames,
                        )
            if obj["name"] == "CausualityOccluder2":
                _, self.frame_list, self.third_party_camera_frames = move_object(
                            self,
                            obj["objectId"],
                            [(0, 0, 0.1),(0, 1, 0)],
                            self.frame_list,
                            self.third_party_camera_frames,
                        )
        #dummy moves for debugging
        self.step("MoveBack", moveMagnitude = 0)
        self.step("MoveAhead", moveMagnitude = 0)

        #count rewards to get output
        self.out = i
        
x = Shape()
x.run()
x.save_frames_to_folder('shape')