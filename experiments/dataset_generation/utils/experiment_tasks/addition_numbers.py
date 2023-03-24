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
        fov=[110,120], # TODO: figure out why 90 FOV bug results in no boxes appearing
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
        max_rewards=3,
        defined_rewards=None,
        num_receptacles=4,
        receptacle_position_limits=[-0.9, 0.9],
            bigPlate="1xPlate",
            smallPlate="0.75xPlate",
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

        assert num_receptacles % 2 == 0, "num_receptacles must be even"
        all_positions = np.linspace(
            *receptacle_position_limits[::-1], num=num_receptacles + 1
        )
        positions = np.delete(all_positions, len(all_positions) // 2)
        occ_positions = [0.6, -0.6]
        defined_rewards = (
            [np.random.randint(0, max_rewards) if position != 0 else np.random.randint(1, max_rewards) for position
             in positions]
            if defined_rewards is None
            else np.array(defined_rewards)
        )
        addition_reward = np.random.randint(1, max_rewards)
        max_reward = np.max(defined_rewards)
        max_defined_rewards = np.where(defined_rewards == np.max(defined_rewards))[
            0
        ]

        if len(max_defined_rewards) != 1:
            selected_max = np.random.choice(max_defined_rewards, 1)
            for selected_reward in max_defined_rewards:
                if selected_reward == selected_max:
                    if max_reward == 0:
                        defined_rewards[selected_reward] = 1
                else:
                    if max_reward != 0:
                        defined_rewards[selected_reward] -= 1

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
            if smallPlate in obj["name"]:
                # left plate (z < 0)
                for position in all_positions:
                    if position == 0:
                        continue
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": 0.0, "y": 0, "z": 0},
                            "position": {"x": -0.2, "y": 1.12, "z": position},
                        }
                    )

            if obj["name"] == "Occluder":

                # left plate (z < 0)
                for position in occ_positions:
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": -0.0, "y": 0, "z": 0},
                            "position": {"x": -0.65, "y": 0.1, "z": position},
                        }
                    )



            # Set the rewards'locations randomly around the plate
            if obj["objectType"] == rewardType:

                for i in range(0, addition_reward):
                    initialPoses.append(
                        {
                            "objectName": obj["name"],
                            "rotation": {"x": 0.0, "y": 0, "z": 0},
                            "position": {
                                "x": 0.25 + random.uniform(-0.06, 0.06),
                                "y": 1.2 + 0.001 * i,
                                "z": random.uniform(-0.13, 0.13),
                            },
                        }
                    )
                # left plate
                for ix, position in enumerate(positions):
                    for i in range(0, defined_rewards[ix]):
                        initialPoses.append(
                            {
                                "objectName": obj["name"],
                                "rotation": {"x": 0.0, "y": 0, "z": 0},
                                "position": {
                                    "x": -0.2 + random.uniform(-0.06, 0.06),
                                    "y": 1.15 + 0.001 * i,
                                    "z": position + random.uniform(-0.13, 0.13),
                                },
                            }
                        )

            # Put lids on 3 plates
            if obj["name"] == "BigBowl":
                # mid plate
                initialPoses.append(
                    {
                        "objectName": bigPlate,
                        "rotation": {"x": 0, "y": 0, "z": 0},
                        "position": {"x": 0.35, "y": 1.11, "z": 0},
                    }
                )
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": 0.25, "y": 0, "z": 180},
                        "position": {"x": 0.2, "y": 1.505, "z": 0},
                    }
                )


            initialPoses.append(initialPose)
        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        current_objects = self.last_event.metadata["objects"].copy()
        # set aside all occluders

        for obj in current_objects:
            if "Occluder" in obj["name"]:

                # left and right stay on table
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [(0, 0, 4), (0.05, 0, 0), (0, 0, -2)],
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
        move_side = np.random.randint(0, len(positions)-1)


        for obj in current_objects:
            if obj["name"].startswith(rewardType) and obj["position"]["x"] >= 0:
                _, self.frame_list, self.third_party_camera_frames = move_object(
                    self,
                    obj["objectId"],
                    [
                        (0, 0, 0.4),
                        (-0.45, 0, 0),
                        (0, -1 * positions[move_side], 0),
                        (0, 0, -0.4),
                    ],
                    self.frame_list,
                    self.third_party_camera_frames,
                )

        if self.last_event.metadata["errorMessage"]:
            print(f'ERROR1:{self.last_event.metadata["errorMessage"]}')
        # count rewards to get output
        defined_rewards[move_side] += addition_reward
        out = np.argmax(defined_rewards)
        self.stats.update(
            {
                "reward_type": rewardType,
                "defined_rewards": defined_rewards,
                "move_plate": move_side,
                "final_greater_plate": int(out),
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
