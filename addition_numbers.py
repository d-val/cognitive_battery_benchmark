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

rewardTypes = ["Potato", "Tomato", "Apple"]

rewardType = random.sample(rewardTypes, 1)[0]


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

    #Set the Plates location (pre-determined)
    if obj["objectType"] == "Plate":
        #right plate (z < 0)
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                    "position": {'x': -0.34, 'y': 1.105, 'z': -0.78}
                    }
                    )

        #left plate (z > 0)
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                    "position": {'x': -0.34, 'y': 1.105, 'z': 0.78}
                    }
                    )
    if obj["name"] == "BigBowl":
        #right bowl (z < 0)
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                    "position": {'x': -0.34, 'y': 1.1, 'z': -0.23}
                    }
                    )

        #left bowl (z > 0)
        initialPoses.append(
                    {"objectName": obj["name"],
                    "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                    "position": {'x': -0.34, 'y': 1.1, 'z': 0.23}
                    }
                    )
    
    #Set the rewards'locations randomly around the plate
    if obj["objectType"] == rewardType:
        #right plate
        for i in range(0,random.randint(0,8)):
            initialPoses.append(
                        {"objectName": obj["name"],
                        "rotation": {'x': 0.0, 'y': 0, 'z': 0},
                        "position": {'x': -0.34 + random.uniform(-0.13, 0.13), 'y': 1.3 + 0.001*i, 'z': -0.75 + random.uniform(-0.13, 0.13)}
                        }
                        )
        #left plate
        for i in range(0,random.randint(0,8)):
            initialPoses.append(
                        {"objectName": obj["name"],
                        "rotation": {'x': 0.0, 'y': 0, 'z': 0},
                        "position": {'x': -0.34 + random.uniform(-0.13, 0.13), 'y': 1.3 + 0.001*i, 'z': 0.75 + random.uniform(-0.13, 0.13)}
                        }
                        )

    #Ignore reward and receptacles object, they will not be randomized on the table
    if obj["objectType"] in {"Pot", rewardType}:
        pass
    elif not obj["moveable"] and not obj["pickupable"]:
        pass
    else:
        initialPoses.append(initialPose)



#set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
controller.step(
  action='SetObjectPoses',
  objectPoses = initialPoses,
  placeStationary=False
)


#Store all rewards Id in list to be exclude from randomization
excludedRewardsId = []
for obj in controller.last_event.metadata["objects"]:
    if obj["objectType"] == rewardType:
        excludedRewardsId.append(obj["objectId"]) #useful for randomization of non-rewards

#randomize all non-rewards objects
controller.step(action="InitialRandomSpawn",
    randomSeed=random.randint(0,10),
    forceVisible=True,
    numPlacementAttempts=5,
    placeStationary=True,
    numDuplicatesOfType = [
    ],
    excludedObjectIds= excludedRewardsId

)

#count rewards to get output
out = None
left = 0
right = 0

controller.step(
    action="PickupObject",
    objectId='Plate|-00.34|+01.11|+00.78',
    forceAction=True,
    manualInteract=True
)

controller.step(
    action="MoveHeldObject",
    ahead=0,
    right= 0,
    up=0.7,
    forceVisible=False
)

controller.step(
    action="MoveHeldObject",
    ahead=0,
    right= 0.5,
    up=0,
    forceVisible=False
)

controller.step(
    action="RotateHeldObject",
    pitch=180,
    yaw=0,
    roll=0
)

for obj in controller.last_event.metadata["objects"]:
    if obj["objectType"] == rewardType:
        if obj["position"]["z"] > 0:
            left += 1
        if obj["position"]["z"] < 0:
            right +=1
if left > right:
    out = -1
elif left < right:
    out = 1
else:   #left == right
    out = 0

print(out)

controller.step("MoveBack")
controller.step("MoveAhead")
controller.step("MoveBack")
controller.step("MoveAhead")
