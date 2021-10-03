# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
from util import *
import random
#unity directory

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

img_array = []

num_of_agents = 1
controller = Controller(
    #local build
    local_executable_path=f"{BASE_DIR}/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
    
    agentMode="default",
    visibilityDistance=2,
    scene="FloorPlan1",

    # step sizes
    gridSize=0.25,
    snapToGrid=False,
    rotateStepDegrees=90,

    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,

    # # camera properties
    width=2000,
    height=2000,
    fieldOfView=random.randint(90,120),
    agentCount = num_of_agents,
    makeAgentsVisible = False
)

#Move agents to fit the screen
for i in range(num_of_agents):
    controller.step(
        action="Teleport",
        position=dict(x=-1.5, y=0.9, z=0),
        rotation=dict(x=0, y=90, z=0),
        horizon=0,
        standing=True,
        agentId = i
    )


#Randomize Materials in the scene
controller.step(
    action="RandomizeMaterials")

#Randomize Lighting in the scene
controller.step(
    action="RandomizeLighting",
    brightness=(0.5, 1.5),
    randomizeColor=True,
    hue=(0, 1),
    saturation=(0.5, 1),
    synchronized=False
)

#Possible receptacle types
receptableTypes = ["Pot", "Mug", "Cup"]

#Randomly chose a receptacle type
receptableType = random.sample(receptableTypes, 1)[0]
receptableType = 'Cup'

#Possible reward objects (Egg, Ball, ...) types
rewardTypes = ["Egg", "Potato", "Tomato", "Apple"]

#Randomly chose a reward type
rewardType = random.sample(rewardTypes, 1)[0]
rewardType = "Egg"
#List of initial poses (receptacle_names' poses)
initialPoses = []
#A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
#https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

excludeList = []                #Egg and Pot exclude from randomization
randomObjects = []              #store all other Pickupable objects

#Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
#and will be location randomized later
for obj in controller.last_event.metadata["objects"]:

    #current Pose of the object
    initialPose = {"objectName": obj["name"],
                      "position": obj["position"],
                      "rotation": obj["rotation"]}

    #Set reward inital position (pre-determined) to the right of the table
    if obj["objectType"] == rewardType :
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                    "position": {'x': -0.43, 'y': 1.126484751701355, 'z': -0.9}
                    }
                    )

    if obj["name"] == "Cup_Opaque":
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 180},
                    "position": {'x': 0, 'y': 1.4, 'z': -0.9}
                    }
                    )
    #Set recetacles location, initialize 3 times on the table at pre-determined positions
    if obj["objectType"] == receptableType and obj["name"] != "Cup_Opaque":
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 180},
                    "position": {'x': -0.43, 'y': 1.5, 'z': 0.5}
                    }
                    )
        initialPoses.append({"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 180},
                    "position": {'x': -0.43, 'y': 1.5, 'z': -7.855288276914507e-05}
                    }
                    )
        initialPoses.append({"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 180},
                    "position": {'x': -0.43, 'y': 1.5, 'z': -0.5}
                    }
                    )
    #Ignore reward and receptacles object, they will not be randomized place behind the table
    if obj["objectType"] in [rewardType] + receptableTypes:
        pass
    else:
        initialPoses.append(initialPose)



#set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
controller.step(
  action='SetObjectPoses',
  objectPoses = initialPoses,
  placeStationary=False
)

#exclude the chosen reward and receptacles from location randomization, 
# only randomize pickupable objects
for obj in controller.last_event.metadata["objects"]:
    if obj["objectType"] in [rewardType] + receptableTypes:
        excludeList.append(obj["objectId"])
    elif obj["pickupable"]:
        randomObjects.append(obj["objectId"])


#exclude all but 1 random objects to show randomly on the table
excludeRandomObjects = random.sample(randomObjects, len(randomObjects)-1)
excludeRandomObjects = []
#https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-randomseed
#InitialRandomSpawn attempts to randomize the position of Pickupable objects, placing them in any valid receptacle they could be placed in within the scene. 
controller.step(action="InitialRandomSpawn",
    randomSeed=random.randint(0,10),
    forceVisible=True,
    numPlacementAttempts=5,
    placeStationary=True,
    numDuplicatesOfType = [

    ],

    #Objects could randomly spawn in any suitable receptacles except for the simulating receptacles themselves
    excludedReceptacles= [receptableType],

    excludedObjectIds= excludeList + excludeRandomObjects

)


#CONSTANTS
MOVEUP_MAGNITUDE = 0.3
MOVE_RECEP_AHEAD_MAG = 0.3

#receptable z coordinate
receptacle_zs = []

#receptable name
receptacle_names = []

#receptacle ids
receptacle_ids = []

#get the z coordinates of the rewardId (Egg) and receptables (Pot) and also get the receptable ids
#get opaque cup Id and x coordinate
for obj in controller.last_event.metadata["objects"]:
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

#sample 1 random receptable to put the (opaque cup + egg) compound under
correct_receptacle_z = random.sample(receptacle_zs,1)[0]

#Calculate how much the egg should be moved to the left to be on top of the intended Pot
egg_move_left_mag = correct_receptacle_z - egg_z

#Calculate how much to move the opaque back to be on the egg
receptacle_move_back = cupOpaque_x - egg_x

print(receptacle_ids)
#move the opaque cup onto the egg
move_object(controller, cupOpaqueId, [(0,0, MOVEUP_MAGNITUDE), (-receptacle_move_back, 0, 0)])

controller.step(
    action="PickupObject",
    objectId=cupOpaqueId,
    forceAction=True,
    manualInteract=True
)

controller.step(
    action="MoveHeldObject",
    ahead=0.3,
    right=0,
    up=0,
    forceVisible=False
)

controller.step("MoveBack")

controller.step("MoveBack")

controller.step("MoveBack")

# move_object(controller, cupOpaqueId, [(receptacle_move_back,0, 0)])




#Swap 2 receptables
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

#get egg final z coordinates
for obj in controller.last_event.metadata["objects"]:
    if obj["objectType"] == rewardType:
        egg_final_z = obj["position"]["z"]

out = None
#determine which pot egg finally in.
#0 = left, 1 = middle, 2 = right
if egg_final_z > -1 and egg_final_z < -0.35:
    out = 2
elif egg_final_z >= -0.35 and egg_final_z <= 0.35:
    out = 1
elif egg_final_z > 0.35 and egg_final_z < 1:
    out = 0

print(out)
#dummy moves for debugging purposes
controller.step("MoveBack")

#for rendering cv2 image
# for i,e in enumerate(multi_agent_event.events):
#     cv2.imshow('agent%s' % i, e.cv2img)



