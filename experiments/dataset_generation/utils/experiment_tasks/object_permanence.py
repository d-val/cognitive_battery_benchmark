# -*- coding: utf-8 -*-
import os
import random

# unity directory
from .utils.experiment import Experiment
from .utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class ObjectPermanence(Experiment):
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
        visibilityDistance=2,
        moveup_magnitude=0.3,
        move_recep_ahead_mag=0.3,
        seed=0,
    ):
        img_array = []
        self.MOVEUP_MAGNITUDE = moveup_magnitude
        self.MOVE_RECEP_AHEAD_MAG = move_recep_ahead_mag
        if type(fov) == list:
            fov = random.randint(*fov)
        super().__init__(
            {
                **{
                    # local build
                    "visibilityDistance": visibilityDistance
                    if type(visibilityDistance) != list
                    else random.randint(*visibilityDistance),
                    # camera properties
                    "fieldOfView": fov if type(fov) != list else random.randint(*fov),
                },
                **controller_args,
            }
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
            if object["objectType"] == rewardType:
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 0},
                        "position": {"x": -0.43, "y": 1.126484751701355, "z": -0.9},
                    }
                )

            if object["name"] == "Cup_Opaque":
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
                        "position": {"x": -0.43, "y": 1.5, "z": 0.5},
                    }
                )
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.43, "y": 1.5, "z": -7.855288276914507e-05},
                    }
                )
                initialPoses.append(
                    {
                        "objectName": object["name"],
                        "rotation": {"x": -0.0, "y": 0, "z": 180},
                        "position": {"x": -0.43, "y": 1.5, "z": -0.5},
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
            if obj["objectType"] == rewardType:
                rewardId = obj["objectId"]
                egg_z = obj["position"]["z"]
                egg_x = obj["position"]["x"]
            if obj["objectType"] == receptableType and obj["name"] != "Cup_Opaque":
                receptacle_names.append(obj["name"])
                receptacle_ids.append(obj["objectId"])
                receptacle_zs.append(obj["position"]["z"])
            if obj["name"] == "Cup_Opaque":
                cupOpaqueId = obj["objectId"]
                cupOpaque_x = obj["position"]["x"]

        # sample 1 random receptable to put the (opaque cup + egg) compound under
        correct_receptacle_z = random.sample(receptacle_zs, 1)[0]

        # Calculate how much the egg should be moved to the left to be on top of the intended Pot
        egg_move_left_mag = correct_receptacle_z - egg_z

        # Calculate how much to move the opaque back to be on the egg
        receptacle_move_back = cupOpaque_x - egg_x

        print(receptacle_ids)
        # move the opaque cup onto the egg
        move_object(
            self,
            cupOpaqueId,
            [(0, 0, self.MOVEUP_MAGNITUDE), (-receptacle_move_back, 0, 0)],
        )

        self.step(
            action="PickupObject",
            objectId=cupOpaqueId,
            forceAction=True,
            manualInteract=True,
        )

        self.step(action="MoveHeldObject", ahead=0.3, right=0, up=0, forceVisible=False)

        self.step("MoveBack")

        self.step("MoveBack")

        self.step("MoveBack")

        # move_object(controller, cupOpaqueId, [(receptacle_move_back,0, 0)])

        # Swap 2 receptables
        # def swap(swap_receptables):
        #   """ swap_receptables: list of 2 receptacle_names object to swap
        #   return None
        #   """
        #   event = controller.last_event
        #   recep1_name = swap_receptables[0]
        #   recep2_name = swap_receptables[1]
        #   recep1_id = get_objectId(recep1_name, controller)
        #   recep2_id = get_objectId(recep2_name, controller)

        #   #calculate the z-different to move the receps
        #   z_different = get_object(recep1_name, controller)["position"]["z"] - get_object(recep2_name, controller)["position"]["z"]

        #   #move first recep far away
        #   move_object(controller, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)])
        # #   img_array.append(controller.last_event.frame)

        #   #move 2nd recep to 1st recep place
        #   move_object(controller, recep2_id, [(0, 0, MOVEUP_MAGNITUDE), (0, -z_different, 0)])
        # #   img_array.append(controller.last_event.frame)

        #   # every time an object is moved, its id is changed
        #   # update 1st receptable ID
        #   recep1_id = get_objectId(recep1_name, controller)

        #   #move 1st recep to second recep place
        #   move_object(controller, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (0, z_different, 0), (-MOVE_RECEP_AHEAD_MAG, 0, 0)])

        # #   img_array.append(controller.last_event.frame)

        # for i in range(random.randint(1,10)):
        #     swap(random.sample(receptacle_names,2))

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

        # for rendering cv2 image
        # for i,e in enumerate(multi_agent_event.events):
        #     cv2.imshow('agent%s' % i, e.cv2img)
