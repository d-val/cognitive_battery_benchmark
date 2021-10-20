# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
import random

# unity directory
from experiment import Experiment
from utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class ObjectPermanence(Experiment):
    def __init__(self, moveup_magnitude=0.3, move_recep_ahead_mag=0.3, seed=0):
        img_array = []
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
                "fieldOfView": random.randint(90, 120),
                "makeAgentsVisible": False,
            }
        )

        #add 3rd party camera
        self.step(
            action="AddThirdPartyCamera",
            position=dict(x=-1.5, y=1, z=0),
            rotation=dict(x=0, y=90, z=0),
            fieldOfView=90,
        )

        # Move agents to fit the screen
        self.step(
            action="Teleport",
            position=dict(x=-1.5, y=0.9, z=0),
            rotation=dict(x=0, y=90, z=0),
            horizon=0,
            standing=True,
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
        receptableType = "Cup"

        # Possible reward objects (Egg, Ball, ...) types
        rewardTypes = ["Egg", "Potato", "Tomato", "Apple"]

        # Randomly chose a reward type
        rewardType = random.sample(rewardTypes, 1)[0]
        # rewardType = "Egg"
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

            # Set reward inital position (pre-determined) to the right of the table
            # if object["objectType"] == rewardType:
            if object["name"] == "Cup_Opaque":
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.43, "y": 1.126484751701355, "z": -0.9},
                    }
                )

            # if object["name"] == "Cup_Opaque":
            if object["objectType"] == rewardType:
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": 0, "y": 1.4, "z": -0.9},
                    }
                )
            # Set recetacles location, initialize 3 times on the table at pre-determined positions
            if (
                object["objectType"] == receptableType
                and object["name"] != "Cup_Opaque"
            ):
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.2, "y": 1.5, "z": 0.5},
                    }
                )
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.2, "y": 1.5, "z": -7.855288276914507e-05},
                    }
                )
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.2, "y": 1.5, "z": -0.5},
                    }
                )
            # Ignore reward and receptacles object, they will not be randomized place behind the table
            if object["objectType"] in [rewardType] + receptableTypes:
                pass
            else:
                initialPoses.append(initialPose)

        # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action="SetObjectPoses", objectPoses=initialPoses, placeStationary=False
        )

        # exclude the chosen reward and receptacles from location randomization,
        # only randomize pickupable objects
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] in [rewardType] + receptableTypes:
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
        receptacle_zs = []

        # receptable name
        receptacle_names = []

        # receptacle ids
        receptacle_ids = []

        # get the z coordinates of the rewardId (Egg) and receptables (Pot) and also get the receptable ids
        # get opaque cup Id and x coordinate
        for obj in self.last_event.metadata["objects"]:
            # if obj["objectType"] == rewardType:
            if obj["name"] == "Cup_Opaque":
                cupOpaqueId = obj["objectId"]
                cupOpaque_z = obj["position"]["z"]
                cupOpaque_x = obj["position"]["x"]
            if obj["objectType"] == receptableType and obj["name"] != "Cup_Opaque":
                receptacle_names.append(obj["name"])
                receptacle_ids.append(obj["objectId"])
                receptacle_zs.append(obj["position"]["z"])
            # if obj["name"] == "Cup_Opaque":
            if obj["objectType"] == rewardType:
                rewardId = obj["objectId"]
                egg_x = obj["position"]["x"]

        # sample 1 random receptable to put the (opaque cup + egg) compound under
        correct_receptacle_z = random.sample(receptacle_zs, 1)[0]

        # Calculate how much the egg should be moved to the left to be on top of the intended Pot
        egg_move_left_mag = correct_receptacle_z - cupOpaque_z

        # Calculate how much to move the opaque back to be on the egg
        receptacle_move_back =  egg_x - cupOpaque_x

        print(receptacle_ids)
        # move the opaque cup onto the egg
        move_object(
            self,
            rewardId,
            [(0, 0, self.MOVEUP_MAGNITUDE), (-receptacle_move_back, 0, 0)],
            self.frame_list,
            self.third_party_camera_frames
        )


        move_object(
            self,
            cupOpaqueId,
            [(0, -egg_move_left_mag, 0)],
            self.frame_list,
            self.third_party_camera_frames
        )

        # self.step(action="MoveHeldObject", ahead=0, right=0.3, up=0, forceVisible=False)

        self.step("MoveBack")

        self.step("MoveBack")

        self.step("MoveBack")

        
        # get egg final z coordinates
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == rewardType:
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

        print(out)
        # dummy moves for debugging purposes
        self.step("MoveBack")

        # for rendering cv2 image
        # for i,e in enumerate(multi_agent_event.events):
        #     cv2.imshow('agent%s' % i, e.cv2img)
x = ObjectPermanence()
x.run()