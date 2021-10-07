# -*- coding: utf-8 -*-
import os
import random

from PIL import Image
from tqdm import tqdm

# unity directory
from experiment import Experiment
from utils.util import get_objectId, get_object, move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class SimpleSwap(Experiment):
    def __init__(self, moveup_magnitude=0.3, move_recep_ahead_mag=0.3, seed=0):

        self.frame_list = []
        self.saved_frames = []
        self.third_party_camera_frames = []

        self.MOVEUP_MAGNITUDE = moveup_magnitude
        self.MOVE_RECEP_AHEAD_MAG = move_recep_ahead_mag

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
                "fieldOfView": random.randint(90, 140),
            },
            seed,
        )

        self.step(
            action="AddThirdPartyCamera",
            position=dict(x=-1.5, y=1, z=0),
            rotation=dict(x=0, y=90, z=0),
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

        # Possible receptacle types
        receptableTypes = ["Pot", "Mug", "Cup"]

        # Randomly chose a receptacle type
        receptableType = random.sample(receptableTypes, 1)[0]

        # Possible reward objects (Egg, Ball, ...) types
        rewardTypes = ["Egg", "Potato", "Tomato", "Apple"]

        # Randomly chose a reward type
        self.rewardType = random.sample(rewardTypes, 1)[0]

        # List of initial poses (Pots' poses)
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

            # Set reward inital position (pre-determined) to the right of the table
            if obj["objectType"] == self.rewardType:
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {
                            "x": -0.4300207197666168,
                            "y": 1.126484751701355,
                            "z": -0.9255303740501404,
                        },
                    }
                )

            # Set recetacles location, initialize 3 times on the table at pre-determined positions
            if obj["objectType"] == receptableType:
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {
                            "x": -0.4351297914981842,
                            "y": 1.1031372547149658,
                            "z": 0.7,
                        },
                    }
                )
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {
                            "x": -0.4351317286491394,
                            "y": 1.1031371355056763,
                            "z": -7.855288276914507e-05,
                        },
                    }
                )
                initialPoses.append(
                    {
                        "objectName": obj["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {
                            "x": -0.4351297914981842,
                            "y": 1.1031371355056763,
                            "z": -0.7,
                        },
                    }
                )
            # Ignore reward and receptacles object, they will not be randomized place behind the table
            if obj["objectType"] in [self.rewardType] + receptableTypes:
                pass
            else:
                initialPoses.append(initialPose)

        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(action="SetObjectPoses", objectPoses=initialPoses)

        # exclude the chosen reward and receptacles from location randomization,
        # only randomize pickupable objects
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] in [self.rewardType] + receptableTypes:
                excludeList.append(obj["objectId"])
            elif obj["pickupable"]:
                randomObjects.append(obj["objectId"])

        # exclude all but 1 random objects to show randomly on the table
        excludeRandomObjects = random.sample(randomObjects, len(randomObjects) - 1)
        excludeRandomObjects = []
        # https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-randomseed
        # InitialRandomSpawn attempts to randomize the position of Pickupable objects, placing them in any valid receptacle they could be placed in within the scene.
        self.step(
            action="InitialRandomSpawn",
            randomSeed=random.randint(0, 10),
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True,
            numDuplicatesOfType=[],
            # Objects could randomly spawn in any suitable receptacles except for the simulating receptacles themselves
            excludedReceptacles=[receptableType],
            excludedObjectIds=excludeList + excludeRandomObjects,
        )

        # receptable z coordinate
        pot_zs = []

        # receptable name
        self.pots = []

        # get the z coordinates of the rewardId (Egg) and receptables (Pot) and also get the receptable ids
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                rewardId = obj["objectId"]
                egg_z = obj["position"]["z"]
            if obj["objectType"] == receptableType:
                self.pots.append(obj["name"])
                pot_zs.append(obj["position"]["z"])

        # sample 1 random receptable to put the rewardId (Egg) in
        correct_pot_z = random.sample(pot_zs, 1)[0]

        # Calculate how much the egg should be moved to the left to be on top of the intended Pot
        egg_move_left_mag = correct_pot_z - egg_z

        # Move agent to fit the screen
        self.step("MoveRight")

        # move the reward to the pre-selected receptable then drop it
        _, self.frame_list, self.third_party_camera_frames = move_object(
            self,
            rewardId,
            [
                (0, 0, self.MOVEUP_MAGNITUDE),
                (0, -egg_move_left_mag, 0),
                (0, 0, -self.MOVEUP_MAGNITUDE),
            ],
            self.frame_list,
            self.third_party_camera_frames,
        )
        # self.frame_list.append(self.last_event.frame)

    # Swap 2 receptables
    def swap(self, swap_receptables):
        """swap_receptables: list of 2 pots object to swap
        return None
        """
        event = self.last_event
        recep1_name = swap_receptables[0]
        recep2_name = swap_receptables[1]
        recep1_id = get_objectId(recep1_name, self)
        recep2_id = get_objectId(recep2_name, self)

        # calculate the z-different to move the receps
        z_different = (
                get_object(recep1_name, self)["position"]["z"]
                - get_object(recep2_name, self)["position"]["z"]
        )

        # move first recep far away
        # move_object(self, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)])
        _, self.frame_list, self.third_party_camera_frames = move_object(
            self,
            recep1_id,
            [(0, 0, self.MOVEUP_MAGNITUDE), (self.MOVE_RECEP_AHEAD_MAG, 0, 0)],
            self.frame_list,
            self.third_party_camera_frames,
        )
        # self.play(move_object(self, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)], self.frame_list))
        self.frame_list.append(self.last_event.frame)
        self.third_party_camera_frames.append(
            self.last_event.third_party_camera_frames[0]
        )
        # move 2nd recep to 1st recep place
        _, self.frame_list, self.third_party_camera_frames = move_object(
            self,
            recep2_id,
            [(0, 0, self.MOVEUP_MAGNITUDE), (0, -z_different, 0)],
            self.frame_list,
            self.third_party_camera_frames,
        )
        # self.frame_list.append(self.last_event.frame)

        # every time an object is moved, its id is changed
        # update 1st receptable ID
        recep1_id = get_objectId(recep1_name, self)

        # move 1st recep to second recep place
        _, self.frame_list, self.third_party_camera_frames = move_object(
            self,
            recep1_id,
            [
                (0, 0, self.MOVEUP_MAGNITUDE),
                (0, z_different, 0),
                (-self.MOVE_RECEP_AHEAD_MAG, 0, 0),
            ],
            self.frame_list,
            self.third_party_camera_frames,
        )

        # self.frame_list.append(self.last_event.frame)

    def run(self):
        for i in range(random.randint(1, 3)):
            self.swap(random.sample(self.pots, 2))

        # get egg final z coordinates
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                egg_final_z = obj["position"]["z"]

        out = None
        # determine which pot egg finally in.
        # 0 = left, 1 = middle, 2 = right
        if -1 < egg_final_z < -0.35:
            out = 2
        elif -0.35 <= egg_final_z <= 0.35:
            out = 1
        elif 0.35 < egg_final_z < 1:
            out = 0

        # dummy moves for debugging purposes
        self.step("MoveBack")
        self = self.step("MoveBack")

        print(out)
        return out

    def save_frames_to_file(self, SAVE_DIR):

        if not os.path.isdir(SAVE_DIR):
            os.makedirs(f"{SAVE_DIR}/frames")
            os.makedirs(f"{SAVE_DIR}/video")

        print("num frames", len(self.frame_list))
        height, width, channels = self.frame_list[0].shape

        for i, frame in enumerate(tqdm(self.frame_list)):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/frames/{i}.jpeg")

        print("num frames", len(self.third_party_camera_frames))
        height, width, channels = self.third_party_camera_frames[0].shape

        for i, frame in enumerate(tqdm(self.third_party_camera_frames)):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/video/{i}.jpeg")


vid = SimpleSwap()
vid.run()
vid.save_frames_to_file("test")
