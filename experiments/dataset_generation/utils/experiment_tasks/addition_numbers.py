# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
from collections import namedtuple

import numpy as np

# unity directory
from .utils.experiment import Experiment
from .utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class AdditionNumbers(Experiment):
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
            "width": 300,
            "height": 300,
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
        rewardType = (
            random.sample(rewardTypes, 1)[0] if rewardType is None else rewardType
        )

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
                #
                # # mid occluder
                # initialPoses.append(
                #     {
                #         "objectName": obj["name"],
                #         "rotation": {"x": -0.0, "y": 0, "z": 0},
                #         "position": {"x": 0.4, "y": 1.105, "z": 0},
                #     }
                # )
            reward = namedtuple("reward", ["left", "middle", "right"])

            defined_rewards = (
                reward(*[np.random.randint(0, max_r) for max_r in max_rewards])
                if defined_rewards is None
                else reward(*defined_rewards)
            )

            # randomly spawn between 0 to 9 food on each plate
            if obj["objectType"] == rewardType:
                # left rewards
                for i in range(defined_rewards.left):  # [0,j)
                    theta = 2 * math.pi * i / defined_rewards.left
                    r = random.uniform(0.1, 0.15)
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {
                                "x": -0.25 + r * math.cos(theta),
                                "y": 1.205,
                                "z": 0.6 + r * math.sin(theta),
                            },
                        }
                    )

                # right rewards
                for i in range(defined_rewards.right):  # [0,k)
                    theta = 2 * math.pi * i / defined_rewards.right
                    r = random.uniform(0.1, 0.15)
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {
                                "x": -0.25 + r * math.cos(theta),
                                "y": 1.205,
                                "z": -0.6 + r * math.cos(theta),
                            },
                        }
                    )

                # mid rewards
                for i in range(defined_rewards.middle):  # [0,l)
                    theta = 2 * math.pi * i / defined_rewards.middle
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
            # Put lids on 3 plates
            if obj["name"] == "BigBowl":
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
                dir = -0.75 if obj["position"]["z"] < 0 else 0.35

                # left and right stay on table
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.4), (-0.73, 0, 0), (0, dir*0.2, 0),(0, 0, -0.5)],
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
                    [(0, 0, 0.4), (0.9, 0, 0), (0, 0, -0.5)],
                    self.frame_list,
                    self.third_party_camera_frames,
                )

        # transfer food
        current_objects = self.last_event.metadata["objects"]
        # TODO: weird double movement of a single apple, point out
        # randomly choose left or right plate, pick random multiplier associates to the direction to move food
        # right, multiplier = -1; left, multiplier = 1
        move_side = random.random()
        if move_side < 0.5:
            multiplier = -1
            move_side = "right"
        else:
            multiplier = 1
            move_side = "left"
        for obj in current_objects:
            if obj["name"].startswith(rewardType) and abs(obj["position"]["z"]) < 0.3:
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

        if self.last_event.metadata["errorMessage"]:
            print(f'ERROR1:{self.last_event.metadata["errorMessage"]}')
        # count rewards to get output
        out = "equal"  # left == right
        left = 0
        right = 0

        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == rewardType:
                if obj["position"]["z"] < 0:
                    right += 1
                if obj["position"]["z"] > 0:
                    left += 1
        if left > right:
            out = "left"
        elif left < right:
            out = "right"
        self.stats.update(
            {
                "reward_type": rewardType,
                "defined_left_reward": defined_rewards.left,
                "defined_middle_reward": defined_rewards.middle,
                "defined_right_reward": defined_rewards.right,
                "move_side": move_side,
                "final_greater_side": out,
            }
        )

        self.label = out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdditionNumbers from file")
    parser.add_argument(
        "saveTo",
        action="store",
        type=str,
        help="which folder to save frames to",
    )
    parser.add_argument(
        "--saveFov",
        action="store",
        type=str,
        help="which perspective video to save",
    )
    parser.add_argument(
        "--fov", action="store", default=[90, 120], help="field of view"
    )
    parser.add_argument(
        "--visDist", action="store", default=5, help="visibility distance of camera"
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=0, help="random seed for experiment"
    )

    parser.add_argument(
        "--height", action="store", type=int, default=800, help="height of the frame"
    )
    parser.add_argument(
        "--width", action="store", type=int, default=800, help="width of the frame"
    )

    parser.add_argument(
        "--rewType",
        action="store",
        type=int,
        help="reward type \n Potato = 0\n Tomato = 1\n Apple = 2",
    )
    parser.add_argument(
        "--rewTypes",
        action="store",
        type=list,
        default=["Potato", "Tomato", "Apple"],
        help='list of possible rewards types, such as ["Potato", "Tomato", "Apple"]',
    )
    parser.add_argument(
        "--maxRew",
        action="store",
        type=list,
        default=[6, 6, 6],
        help="maximum rewards across the [left, middle, right] plate",
    )
    parser.add_argument(
        "--defRew",
        action="store",
        type=list,
        help="defined rewards across the [left, middle, right] plate",
    )

    args = parser.parse_args()
    # TODO: add assertion on types and values here, reorder inputs

    experiment = AdditionNumbers(
        {"height": args.height, "width": args.width},
        fov=args.fov,
        visibilityDistance=args.visDist,
        seed=args.seed,
    )

    experiment.run(
        rewardTypes=args.rewTypes,
        rewardType=args.rewType,
        max_rewards=args.maxRew,
        defined_rewards=args.defRew,
    )

    experiment.stop()
    experiment.save_frames_to_folder(args.saveTo, args.saveFov)
