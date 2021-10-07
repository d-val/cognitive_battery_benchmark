# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
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
    def __init__(self, seed=0):

        self.frame_list = []
        self.saved_frames = []
        self.third_party_camera_frames = []

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
                "fieldOfView": random.randint(90, 120),
                "makeAgentsVisible": False,
            },
            seed,
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

        rewardTypes = ["Potato", "Tomato", "Apple"]

        self.rewardType = random.sample(rewardTypes, 1)[0]

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
                # right occluder
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": 0.15, "y": 1.105, "z": -0.6},
                    }
                )
                # left occluder
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

            # randomly spawn between 0 to 9 food on each plate
            if obj["objectType"] == self.rewardType:
                # right rewards
                j = random.randint(0, 6)
                for i in range(j):  # [0,j)
                    theta = 2 * math.pi * i / j
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

                # left rewards
                k = random.randint(0, 6)
                for i in range(k):  # [0,k)
                    theta = 2 * math.pi * i / k
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
                l = random.randint(0, 6)
                for i in range(l):  # [0,l)
                    theta = 2 * math.pi * i / l
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
                print(j, k, l)
            # Put lids on 3 plates
            if obj["name"] == "BigBowl":
                # right plate (z < 0)
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.25, "y": 1.455, "z": -0.6},
                    }
                )

                # left plate (z > 0)
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

        self.step("MoveBack")
        self.step("MoveAhead")
        self.step("MoveBack")
        self.step("MoveAhead")
        # count rewards to get output
        out = None
        left = 0
        right = 0

        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                if obj["position"]["z"] > 0:
                    left += 1
                if obj["position"]["z"] < 0:
                    right += 1
        if left > right:
            out = -1
        elif left < right:
            out = 1
        else:  # left == right
            out = 0
        print(out)

    def save_frames_to_file(self, SAVE_DIR):

        if not os.path.isdir(SAVE_DIR):
            os.makedirs(f"{SAVE_DIR}/addition_numbers_agent")
            os.makedirs(f"{SAVE_DIR}/addition_numbers_monkey")

        print("num frames", len(self.frame_list))
        height, width, channels = self.frame_list[0].shape

        for i, frame in enumerate(tqdm(self.frame_list)):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/addition_numbers_agent/{i}.jpeg")

        print("num frames", len(self.third_party_camera_frames))
        height, width, channels = self.third_party_camera_frames[0].shape

        for i, frame in enumerate(tqdm(self.third_party_camera_frames)):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/addition_numbers_monkey/{i}.jpeg")
