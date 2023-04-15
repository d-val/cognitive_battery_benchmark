# -*- coding: utf-8 -*-
import argparse
import os
import random
from collections import namedtuple

import numpy as np

# unity directory
from .utils.experiment import Experiment

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class RelativeNumbers(Experiment):
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
        visibilityDistance=2,
        seed=0,
    ):
        self.seed = seed
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
            action="Teleport",
            # position=dict(x=-1.5, y=0.9, z=0),
            position=dict(x=-1.5, y=0.9, z=0),
            rotation=dict(x=0, y=90, z=0),
            horizon=0,
            standing=True,
        )
        # print(self.last_event.metadata["agent"]["position"])
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
        rewardType=None,
        rewardTypes=["Potato", "Tomato", "Apple"],
        max_rewards=8,
        defined_rewards=None,
        num_receptacles=4,
        receptacle_position_limits=[-0.9, 0.9],
        plate="0.5xPlate",
        engaged_receptacles=None,
    ):
        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        # A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
        # https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

        excludeList = []  # Egg and Pot exclude from randomization
        randomObjects = []  # store all other Pickupable objects

        rewardType = (
            rewardType if rewardType is not None else random.sample(rewardTypes, 1)[0]
        )

        # Set the Plates location (pre-determined)
        positions = np.linspace(*receptacle_position_limits[::-1], num=num_receptacles)
        if engaged_receptacles is None:
            defined_rewards = (
                [np.random.randint(0, max_rewards + 1) for _ in range(num_receptacles)]
                if defined_rewards is None
                else np.array(defined_rewards)
            )
        else:
            engaged_receptacles_ix = np.random.choice(
                num_receptacles, engaged_receptacles, replace=False
            )
            defined_rewards = [
                np.random.randint(1, max_rewards + 1)
                if ix in engaged_receptacles_ix
                else 0
                for ix in range(num_receptacles)
            ]
        max_reward = np.max(defined_rewards)
        max_defined_rewards = np.where(defined_rewards == np.max(defined_rewards))[0]
        if len(max_defined_rewards) != 1:
            selected_max = np.random.choice(max_defined_rewards, 1)[0]
            for selected_reward in max_defined_rewards:
                if selected_reward == selected_max:
                    if max_reward == 0:
                        defined_rewards[selected_reward] = 1
                else:
                    if max_reward != 0:
                        if engaged_receptacles is None:
                            defined_rewards[selected_reward] -= 1
                        else:
                            defined_rewards[selected_max] += 1
        defined_rewards = np.array(defined_rewards)
        # Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        for object in self.last_event.metadata["objects"]:

            # current Pose of the object
            initialPose = {
                "objectName": object["name"],
                "position": object["position"],
                "rotation": object["rotation"],
            }

            if plate in object["name"]:
                # left plate (z < 0)
                for position in positions:
                    initialPoses.append(
                        {
                            "objectName": object["name"],
                            "rotation": {
                                "x": -0.0,
                                "y": 0,
                                "z": 0,
                            },
                            "position": {
                                "x": -0.34,
                                "y": 1.105,
                                "z": position,
                            },
                        }
                    )

            # Set the rewards'locations randomly around the plate
            if object["objectType"] == rewardType:

                # left plate
                for ix, position in enumerate(positions):
                    for i in range(0, defined_rewards[ix]):
                        initialPoses.append(
                            {
                                "objectName": object["name"],
                                "rotation": {
                                    "x": 0.0,
                                    "y": 0,
                                    "z": 0,
                                },
                                "position": {
                                    "x": -0.34 + random.uniform(-0.13, 0.13),
                                    "y": 1.15 + 0.001 * i,
                                    "z": position + random.uniform(-0.13, 0.13),
                                },
                            }
                        )

            # Ignore reward and receptacles object, they will not be randomized on the table
            if object["objectType"] in {"Plate", rewardType}:
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
            if obj["objectType"] == rewardType:
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

        self.frame_list = [self.last_event.frame]
        if self.depth_list is not None:
            self.depth_list = [self.last_event.depth_frame]
        if self.segmentation_list is not None:
            self.segmentation_list = [self.last_event.instance_segmentation_frame]
        out = np.argmax(defined_rewards)
        self.stats.update(
            {
                "reward_type": rewardType,
                "final_label": defined_rewards.tolist(),
                "final_greater_side": int(out),
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
        default=[8, 8],
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

    experiment = RelativeNumbers(
        {"height": args.height, "width": args.width},
        fov=args.fov,
        visibilityDistance=args.visDist,
        seed=args.seed,
    )
    experiment.run(
        rewardType=args.rewType,
        rewardTypes=args.rewTypes,
        max_rewards=args.maxRew,
        defined_rewards=args.defRew,
    )
    experiment.stop()
    experiment.save_frames_to_folder(args.saveTo, args.saveFov)
