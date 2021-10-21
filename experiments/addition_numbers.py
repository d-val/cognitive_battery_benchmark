# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
import random
from tqdm import tqdm
import math

# unity directory
from experiment import Experiment
from utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class AdditionNumbers(Experiment):
    def __init__(self, fov=None, seed=0):

        random.seed(seed)
        np.random.seed(seed)

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
                "makeAgentsVisible": False,
            }
        )

        self.step(
            action="AddThirdPartyCamera",
            position=dict(x=1.5, y=1.8, z=0),
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

        # #Randomize Materials in the scene
        # controller.step(
        #     action="RandomizeMaterials")

        # #Randomize Lighting in the scene
        # controller.step(
        #     action="RandomizeLighting",
        #     brightness=(0.5, 1.5),
        #     randomizeColor=True,
        #     hue=(0, 1),
        #     saturation=(0.5, 1),
        #     synchronized=False
        # )

    def run(
        self,
        rewardTypes=["Potato", "Tomato", "Apple"],
        rewardType=None,
        max_rewards=[6, 6, 6],
        defined_rewards=None,
    ):
        self.rewardType = random.sample(rewardTypes, 1)[0]

        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        # A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
        # https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

        excludeList = []  # Egg and Pot exclude from randomization
        randomObjects = []  # store all other Pickupable objects

        # Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        for obj in self.last_event.metadata["objects"]:

            # current Pose of the object
            initialPose = {
                "objectName": obj["name"],
                "position": obj["position"],
                "rotation": obj["rotation"],
            }

            # Set the Plates location (pre-determined)
            if obj["objectType"] == "Plate":
                # right plate (z < 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.25, "y": 1.105, "z": -0.6},
                    }
                )

                # left plate (z > 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.25, "y": 1.105, "z": 0.6},
                    }
                )

                # mid plate
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": 0, "y": 1.105, "z": 0},
                    }
                )

            if obj["name"] == "Occluder":
                # left occluder
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": 0.15, "y": 1.105, "z": -0.6},
                    }
                )
                # right occluder
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": 0.15, "y": 1.105, "z": 0.6},
                    }
                )

                # mid occluder
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": 0.4, "y": 1.105, "z": 0},
                    }
                )

            defined_left_reward, defined_mid_reward, defined_right_reward = (
                defined_rewards
                if defined_rewards is not None
                else [np.random.randint(0, max_r) for max_r in max_rewards]
            )

            # randomly spawn between 0 to 9 food on each plate
            if obj["objectType"] == self.rewardType:
                # left rewards
                for i in range(defined_left_reward):  # [0,j)
                    theta = 2 * math.pi * i / defined_left_reward
                    r = random.uniform(0.1, 0.15)
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {
                                "x": -0.25 + r * math.cos(theta),
                                "y": 1.205,
                                "z": -0.6 + r * math.sin(theta),
                            },
                        }
                    )

                # right rewards
                for i in range(defined_right_reward):  # [0,k)
                    theta = 2 * math.pi * i / defined_right_reward
                    r = random.uniform(0.1, 0.15)
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {
                                "x": -0.25 + r * math.cos(theta),
                                "y": 1.205,
                                "z": 0.6 + r * math.cos(theta),
                            },
                        }
                    )

                # mid rewards
                for i in range(defined_mid_reward):  # [0,l)
                    theta = 2 * math.pi * i / defined_mid_reward
                    r = random.uniform(0.1, 0.15)
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {
                                "x": 0 + r * math.cos(theta),
                                "y": 1.205,
                                "z": 0 + +r * math.cos(theta),
                            },
                        }
                    )
                print(
                    f"Left Reward: {defined_left_reward} | Mid Reward: {defined_mid_reward} | Right Reward: {defined_right_reward}"
                )
            # Put lids on 3 plates
            if obj["name"] == "BigBowl":
                # left plate (z < 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.25, "y": 1.455, "z": -0.6},
                    }
                )

                # right plate (z > 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.25, "y": 1.455, "z": 0.6},
                    }
                )

                # mid plate
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0, "y": 1.455, "z": 0},
                    }
                )

            initialPoses.append(initialPose)

        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        current_objects = self.last_event.metadata["objects"]
        # set aside all occluders
        for obj in current_objects:
            if obj["name"][:8] == "Occluder":
                # left and right stay on table
                if abs(obj["position"]["z"]) > 0.3:
                    _, self.frame_list, self.third_party_camera_frames = move_object(
                        self,
                        obj["objectId"],
                        [(0, 0, 0.4), (-0.73, 0, 0), (0, 0, -0.5)],
                        self.frame_list,
                        self.third_party_camera_frames,
                    )
                # middle goes away
                else:
                    _, self.frame_list, self.third_party_camera_frames = move_object(
                        self,
                        obj["objectId"],
                        [(0, 0, 0.4), (-1.2, 0, 0), (0, 0, -0.5)],
                        self.frame_list,
                        self.third_party_camera_frames,
                    )

        current_objects = self.last_event.metadata["objects"]
        # remove all bowls
        for obj in current_objects:
            if obj["name"][:7] == "BigBowl" and abs(obj["position"]["z"]):
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.4), (-0.73, 0, 0), (0, 0, -0.5)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )

        current_objects = self.last_event.metadata["objects"]
        # put sides occluder back
        for obj in current_objects:
            # only put right and left occluders back
            if obj["name"][:8] == "Occluder" and abs(obj["position"]["z"]) > 0.3:
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.4), (+0.73, 0, 0), (0, 0, -0.5)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )

        # transfer food
        current_objects = self.last_event.metadata["objects"]

        # randomly choose left or right plate, pick random multiplier associates to the direction to move food
        # right, multiplier = 1; left, multiplier = -1
        if random.random() > 0.5:
            multiplier = 1
        else:
            multiplier = -1
        for obj in current_objects:
            if (
                obj["name"].startswith(self.rewardType)
                and abs(obj["position"]["z"]) < 0.3
            ):
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [
                        (0, 0, 0.4),
                        (-0.25, 0, 0),
                        (0, -0.6 * multiplier, 0),
                        (0, 0, -0.4),
                    ],
                    self.frame_list,
                    self.third_party_camera_frames,
                )

        # dummy moves for debugging
        self.step("MoveBack", moveMagnitude=0)
        self.step("MoveAhead", moveMagnitude=0)

        # count rewards to get output
        self.out = 0  # left == right
        left = 0
        right = 0

        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                if obj["position"]["z"] > 0:
                    right += 1
                if obj["position"]["z"] < 0:
                    left += 1
        if left > right:
            self.out = -1
        elif left < right:
            self.out = 1
