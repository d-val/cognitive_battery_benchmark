# -*- coding: utf-8 -*-
import argparse
import math
import os
import random

# unity directory
import numpy as np

from .utils.experiment import Experiment
from .utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Rotation(Experiment):
    # set distance of cups to the center of tray (left,middle,right)
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

    def run(
        self,
        case=None,
        distances=None,
        rewardTypes=["Potato", "Tomato", "Apple"],
        rewardType=None,
        degree_rotation_per_frame=9,
        moveup_magnitude=0.4,
    ):
        # List of initial poses (receptacle_names' poses)
        case = case if case is not None else random.randint(1, 3)
        distances = (
            distances
            if distances is not None
            else {"left": 0.4, "middle": 0, "right": -0.4}
        )
        rewardType = (
            random.sample(rewardTypes, 1)[0] if rewardType is None else rewardType
        )

        initialPoses = []
        # A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
        # https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

        excludeList = []  # Egg and Pot exclude from randomization
        randomObjects = []  # store all other Pickupable objects

        # set number of rotation, 11 for 360 degree and 6 for 180 degree
        # 11 means rotate 10 times 36 degree each and 6 means rotate 5 times
        degrees_to_rotate = 0

        # rotate 180, food in middle
        if case == 1:
            degrees_to_rotate = 180
            food_dist = distances["middle"]
        # rotate 360, food in left or right
        if case == 2:
            degrees_to_rotate = 360
            food_dist = random.choice([distances["left"], distances["right"]])
        # rotate 180, food in left or right
        if case == 3:
            degrees_to_rotate = 180
            food_dist = random.choice([distances["left"], distances["right"]])
        # Initialize Object by specifying each object location, receptacle and reward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        assert (
            degrees_to_rotate % degree_rotation_per_frame == 0
        ), "Degrees to rotate must be divisible by degree_rotation_per_frame"
        for i in range(0, int(degrees_to_rotate / degree_rotation_per_frame) + 1):
            # empty initial poses after each iteration to avoid duplicate
            initialPoses = []
            for obj in self.last_event.metadata["objects"]:
                angle = i * degree_rotation_per_frame
                # print(angle, i, degrees_to_rotate)
                angle_radian = 2 * math.pi * angle / 360
                # current Pose of the object
                initialPose = {
                    "objectName": obj["name"],
                    "position": obj["position"],
                    "rotation": obj["rotation"],
                }

                # if obj["name"] == "Tray":
                #     #mid occluder
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': angle, 'z': 0},
                #                 "position": {'x': 0, 'y': 1.105, 'z': 0}
                #                 }
                #                 )
                # if obj["objectType"] == "Potato":
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                #                 "position": {'x': 0 + 0.4 * math.sin(angle_radian), 'y': 1.205, 'z': 0.4*math.cos(angle_radian)}
                #                 }
                #                 )
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                #                 "position": {'x': 0 + 0 * math.sin(angle_radian), 'y': 1.205, 'z': 0*math.cos(angle_radian)}
                #                 }
                #                 )
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                #                 "position": {'x': 0 - 0.4 * math.sin(angle_radian), 'y': 1.205, 'z': -0.4*math.cos(angle_radian)}
                #                 }
                #                 )
                if obj["objectType"] == rewardType:
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": angle, "z": 0},
                            "position": {
                                "x": food_dist * math.sin(angle_radian),
                                "y": 1.205,
                                "z": food_dist * math.cos(angle_radian),
                            },
                        }
                    )

            initialPoses.append(
                {
                    "objectName": "Tray",
                    "rotation": {"x": -0.0, "y": angle, "z": 0},
                    "position": {"x": 0, "y": 1.105, "z": 0},
                }
            )
            initialPoses.append(
                {
                    "objectName": "Cup1",
                    "rotation": {"x": -0.0, "y": angle, "z": 180},
                    "position": {
                        "x": 0 + distances["right"] * math.sin(angle_radian),
                        "y": 1.505,
                        "z": distances["right"] * math.cos(angle_radian),
                    },
                }
            )
            initialPoses.append(
                {
                    "objectName": "Cup1",
                    "rotation": {"x": -0.0, "y": angle, "z": 180},
                    "position": {
                        "x": distances["middle"] * math.sin(angle_radian),
                        "y": 1.505,
                        "z": distances["middle"] * math.cos(angle_radian),
                    },
                }
            )
            initialPoses.append(
                {
                    "objectName": "Cup1",
                    "rotation": {"x": -0.0, "y": angle, "z": 180},
                    "position": {
                        "x": distances["left"] * math.sin(angle_radian),
                        "y": 1.505,
                        "z": distances["left"] * math.cos(angle_radian),
                    },
                }
            )

            # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
            self.step(
                action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
            )

            # add frame to corresponding frame list
            self.frame_list.append(self.last_event.frame)
            self.third_party_camera_frames.append(
                self.last_event.third_party_camera_frames[0]
            )
            # initial state, lift up cup to show food
            if i == 0:
                for obj in self.last_event.metadata["objects"]:
                    if obj["name"][:4] == "Cup1":
                        move_object(
                            self,
                            obj["objectId"],
                            [(0, 0, moveup_magnitude), (0, 0, -moveup_magnitude)],
                            self.frame_list,
                            self.third_party_camera_frames,
                        )
        out = None
        # return value
        # 1 = right, 0 = middle, -1 = left
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == rewardType:
                dist = obj["position"]["z"]
                if dist < 0.3:
                    out = "right"
                if dist > -0.3:
                    out = "left"
                if abs(dist) <= 0.3:
                    out = "middle"

        for loc, val in distances.items():
            if val == food_dist:
                initialLoc = loc

        self.stats.update(
            {
                "case": case,
                "distances": distances,
                "reward_type": rewardType,
                "initial_object_location": initialLoc,
                "final_object_location": out,
            }
        )

        self.label = out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rotation from file")
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
        "--case",
        action="store",
        type=list,
        help="specific rotation case",
    )
    parser.add_argument(
        "--distances",
        action="store",
        type=list,
        help="defined distances for the [left, middle, right] receptacle",
    )

    args = parser.parse_args()
    # TODO: add assertion on types and values here, reorder inputs
    # TODO: update arguments (degrees to rotate and moveupMag)

    experiment = Rotation(
        {"height": args.height, "width": args.width},
        fov=args.fov,
        visibilityDistance=args.visDist,
        seed=args.seed,
    )
    experiment.run(
        case=args.case,
        distances=args.distances,
        rewardTypes=args.rewTypes,
        rewardType=args.rewType,
    )
    experiment.stop()
    experiment.save_frames_to_folder(args.saveTo, args.saveFov)
