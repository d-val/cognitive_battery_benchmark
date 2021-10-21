# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
import random

# unity directory
from experiment import Experiment

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class RelativeNumbers(Experiment):
    def __init__(
        self, rewardType=None, rewardTypes=["Potato", "Tomato", "Apple"], seed=0
    ):
        self.seed = seed
        super().__init__(
            {
                # local build
                "local_executable_path": f"{BASE_DIR}/utils/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
                "agentMode": "default",
                "visibilityDistance": 2,
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
        )
        self.step(
            action="Teleport",
            position=dict(x=-1.5, y=0.9, z=0),
            rotation=dict(x=0, y=90, z=0),
            horizon=0,
            standing=True,
        )

        self.rewardType = (
            rewardType if rewardType is not None else random.sample(rewardTypes, 1)[0]
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

    def run(self, max_rewards=[8, 8], defined_rewards=None):
        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        # A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
        # https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

        excludeList = []  # Egg and Pot exclude from randomization
        randomObjects = []  # store all other Pickupable objects

        # Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        for object in self.last_event.metadata["objects"]:

            # current Pose of the object
            initialPose = {
                "objectName": object["name"],
                "position": object["position"],
                "rotation": object["rotation"],
            }

            # Set the Plates location (pre-determined)
            if object["objectType"] == "Plate":
                # left plate (z < 0)
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.34, "y": 1.105, "z": -0.34},
                    }
                )

                # right plate (z > 0)
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.34, "y": 1.105, "z": 0.34},
                    }
                )

            defined_left_reward, defined_right_reward = (
                defined_rewards
                if defined_rewards is not None
                else [np.random.randint(0, max_r) for max_r in max_rewards]
            )

            # Set the rewards'locations randomly around the plate
            if object["objectType"] == self.rewardType:

                # left plate
                for i in range(0, defined_left_reward):
                    initialPoses.append(
                        {
                            "objectName": object["name"],
                            "rotation": {"x": 0.0, "y": 0, "z": 0},
                            "position": {
                                "x": -0.34 + random.uniform(-0.13, 0.13),
                                "y": 1.15 + 0.001 * i,
                                "z": -0.34 + random.uniform(-0.13, 0.13),
                            },
                        }
                    )
                # right plate
                for i in range(0, defined_right_reward):
                    initialPoses.append(
                        {
                            "objectName": object["name"],
                            "rotation": {"x": 0.0, "y": 0, "z": 0},
                            "position": {
                                "x": -0.34 + random.uniform(-0.13, 0.13),
                                "y": 1.15 + 0.001 * i,
                                "z": 0.34 + random.uniform(-0.13, 0.13),
                            },
                        }
                    )

            # Ignore reward and receptacles object, they will not be randomized on the table
            if object["objectType"] in {"Plate", self.rewardType}:
                pass
            elif not object["moveable"] and not object["pickupable"]:
                pass
            else:
                initialPoses.append(initialPose)

        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        # Store all rewards Id in list to be exclude from randomization
        excludedRewardsId = []
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                excludedRewardsId.append(
                    obj["objectId"]
                )  # useful for randomization of non-rewards

        # randomize all non-rewards objects
        self.step(
            action="InitialRandomSpawn",
            randomSeed=self.seed,
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True,
            excludedObjectIds=excludedRewardsId,
        )

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

        # dummy move for visual
        self.step("MoveBack", moveMagnitude=0)
        self.step("MoveBack", moveMagnitude=0)
