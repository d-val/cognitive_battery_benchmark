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


class RotationChain(Experiment):
    # set distance of cups to the center of tray (left,middle,right)
    def __init__(
        self,
        controller_args={
            "local_executable_path": "utils/test.app/Contents/MacOS/AI2-THOR",
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
        angle=None,
        reward_loc=None,
        distances=None,
        rewardTypes=["Potato", "Tomato", "Apple"],
        rewardType=None,
        degree_rotation_per_frame=9,
        moveup_magnitude=0.4,
        num_receptacles=6,
        receptacle_position_limits=[-0.9, 0.9],
        num_rotations=5,
    ):
        # List of initial poses (receptacle_names' poses)
        distances = (
            distances
            if distances is not None
            else np.linspace(*receptacle_position_limits[::-1], num=num_receptacles)
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
        degrees_to_rotate, reward_loc = 180, np.random.randint(0, num_receptacles - 1)
        init_reward_loc = reward_loc
        rotations = [random.sample(list(zip(range(num_receptacles), range(num_receptacles)[1:])), 1)[0] for _ in range(num_rotations)]
        # Initialize Object by specifying each object location, receptacle and reward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        assert (
            degrees_to_rotate % degree_rotation_per_frame == 0
        ), "Degrees to rotate must be divisible by degree_rotation_per_frame"

        for obj in self.last_event.metadata["objects"]:
            initialPose = {
                "objectName": obj["name"],
                "position": obj["position"],
                "rotation": obj["rotation"],
            }

            if obj["objectType"] == rewardType:
                pos_x, pos_z = 0, distances[reward_loc]
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {
                            "x": pos_x,
                            "y": 1.205,
                            "z": pos_z
                        },
                    }
                )
            if obj["objectType"] == "Cup":
                for ix, position in enumerate(distances):
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": 0.0, "y": 0, "z": 180},
                            "position": {
                                "x": 0,
                                "y": 1.505,
                                "z": position,
                            },
                        }
                    )

        # initialPoses.append(
        #     {
        #         "objectName": "Tray",
        #         "rotation": {"x": -0.0, "y": 0, "z": 0},
        #         "position": {"x": 0, "y": 1.105, "z": 0},
        #     }
        # )


        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        # add frame to corresponding frame list
        self.update_frames()
        # initial state, lift up cup to show food

        for obj in self.last_event.metadata["objects"]:
            if "Cup" in obj["name"]:
                move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, moveup_magnitude), (0, 0, -moveup_magnitude)],
                )
        for obj in self.last_event.metadata["objects"]:
            if "Cup" in obj["name"]:
                lowest_position = obj["position"]["y"]
                break
        for ix_rot, (obj1, obj2) in enumerate(rotations):
            mid_point = (distances[obj1] + distances[obj2]) / 2
            for i in range(0, int(degrees_to_rotate / degree_rotation_per_frame) + 1):
                # empty initial poses after each iteration to avoid duplicate
                initialPoses = []
                angle = i * degree_rotation_per_frame
                # print(angle, i, degrees_to_rotate)
                angle_radian = 2 * math.pi * angle / 360



                for ix, position in enumerate(distances):
                    if ix in {obj1, obj2}:
                        initialPoses.append(
                            {
                                "objectName": "Cup",
                                "rotation": {"x": -0.0, "y": angle, "z": 180},
                                "position": {
                                    "x": (position - mid_point) * math.sin(angle_radian),
                                    "y": lowest_position,
                                    "z": ((position - mid_point) * math.cos(angle_radian)) + mid_point,
                                },
                            }
                        )
                    else:
                        initialPoses.append(
                            {
                                "objectName": "Cup",
                                "rotation": {"x": -0.0, "y": angle, "z": 180},
                                "position": {
                                    "x": 0,
                                    "y": lowest_position,
                                    "z": position,
                                },
                            }
                        )


                # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
                self.step(
                    action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
                )

                # add frame to corresponding frame list
                self.update_frames()

            initialPoses = []
            for ix, position in enumerate(distances):
                initialPoses.append(
                    {
                        "objectName": "Cup",
                        "rotation": {"x": 0.0, "y": 0, "z": 180},
                        "position": {
                            "x": 0,
                            "y": lowest_position,
                            "z": position,
                        },
                    }
                )
            self.step(
                action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
            )

            self.update_frames()

        out = determine_final_loc(rotations, init_reward_loc, len(distances))
        # for loc, val in distances.items():
        #     if val == food_dist:
        #         initialLoc = loc

        self.stats.update(
            {
                "degrees_to_rotate": int(degrees_to_rotate),
                "distances": distances.tolist(),
                "rotations": {f'rotation_{ix}': value for ix, value in enumerate(rotations)},
                "reward_type": rewardType,
                "initial_object_location": init_reward_loc,
                "final_label": int(out),
            }
        )
        self.label = out


def determine_final_loc(rotations, rewardLoc, numReceptacles):
    reward_arr = np.zeros(numReceptacles)
    reward_arr[rewardLoc] = 1
    for obj1, obj2 in rotations:
        reward_arr[[obj2, obj1]] = reward_arr[[obj1, obj2]]

    return np.where(reward_arr == 1)[0][0]

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
