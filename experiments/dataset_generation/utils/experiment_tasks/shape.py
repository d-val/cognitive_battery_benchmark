# -*- coding: utf-8 -*-
import argparse
import os
import random

import numpy as np

# unity directory
from .utils.experiment import Experiment
from .utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Shape(Experiment):
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
        rewardTypes=["Potato", "Tomato", "Apple"],
        rewardType=None,
        coveringTypes=["0.75xPlate"],
        coveringType=None,
        receptacle_position_limits=[-0.9, 0.9],
        num_receptacles=4,
        engaged_receptacles=2,
    ):
        # TODO: add ability to specify number of items in each plate
        self.rewardType, self.coveringType = (
            random.sample(rewardTypes, 1)[0] if rewardType is None else rewardType,
            random.sample(coveringTypes, 1)[0]
            if coveringType is None
            else coveringType,
        )

        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        afterPoses = []

        positions = np.linspace(
            *receptacle_position_limits[::-1], num=num_receptacles
        )
        # randomly sample between 0 and 1 for each position
        if engaged_receptacles is None:
            defined_rewards = np.random.random(
                len(positions)
            )  # TODO: implement defined_rewards
        else:
            defined_rewards = np.zeros(len(positions))
            defined_rewards[:engaged_receptacles] = 1
            np.random.shuffle(defined_rewards)

        occ_positions = [0.4, -0.4]

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
            if self.coveringType in obj["name"]:
                # right Cardboard1 (z > 0)
                for position in positions:
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 180},
                            "position": {"x": 0, "y": 1.305, "z": position},
                        }
                    )

            if obj["name"] == "Occluder":

                # left plate (z < 0)
                for position in occ_positions:
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {"x": -0.6, "y": 1.2, "z": position},
                        }
                    )

            elif obj["objectType"] == self.rewardType:
                for position, valid in zip(positions, defined_rewards):
                    if valid > 0.5:
                        initialPoses.append(
                            {
                                "objectName": obj["name"],
                                "rotation": {"x": -0.0, "y": 0, "z": 0},
                                "position": {"x": 0.012, "y": 1.15, "z": position},
                            }
                        )

            else:
                initialPoses.append(initialPose)

        # set initial Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        for obj in self.last_event.metadata["objects"]:
            if obj["name"] == "Occluder":
                move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.5), (0.95, 0, 0), (0, 0, -0.5)],
                )
                break
            elif "Occluder" in obj["name"]:
                move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 0.5), (0.95, 0, 0), (0, 0, -0.5)],
                )
                break

        if self.last_event.metadata["errorMessage"]:
            print(f'ERROR1:{self.last_event.metadata["errorMessage"]}')
        # count rewards to get output

        self.stats.update(
            {
                "final_label": np.where(defined_rewards > 0.5)[0].tolist(),
                "reward_locs": np.where(defined_rewards > 0.5, 1, 0).tolist(),
            }
        )
        self.label = defined_rewards.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Shape from file")
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
        "--covType",
        action="store",
        type=int,
        help="a specific covering type in ['Plate','Bowl']",
    )
    parser.add_argument(
        "--covTypes",
        action="store",
        type=list,
        default=["Plate", "Bowl"],
        help='list of possible covering types, such as ["Plate","Bowl"]',
    )

    args = parser.parse_args()
    # TODO: add assertion on types and values here, reorder inputs

    experiment = Shape(
        {"height": args.height, "width": args.width},
        fov=args.fov,
        visibilityDistance=args.visDist,
        seed=args.seed,
    )

    experiment.run(
        rewardTypes=args.rewTypes,
        rewardType=args.rewType,
        coveringTypes=args.covTypes,
        coveringType=args.covType,
    )

    experiment.stop()
    experiment.save_frames_to_folder(args.saveTo, args.saveFov)
