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

class VideoBenchmark(Controller):
    
    def __init__(self):
        import argparse

        msg = "Adding description"
        # Initialize parser
        my_parser = argparse.ArgumentParser(description = msg)

        my_parser.add_argument('--height', action='store', type=int, help = "height of the frame")
        my_parser.add_argument('--width', action='store', type=int, help = "width of the frame")
        my_parser.add_argument('-r', '--reward', action='store', type=int, help = "reward type \n Potato = 0\n Tomato = 1\n Apple = 2")
        my_parser.add_argument('--left', action='store', type=int, help = "enter number of rewards on left plate\n")
        my_parser.add_argument('--right', action='store', type=int, help = "enter number of rewards on right plate \n")

        args = my_parser.parse_args()
        super().__init__(
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
            width=args.width if args.width != None else 2000,
            height=args.height if args.height != None else 2000,
            fieldOfView=random.randint(90,120),
            makeAgentsVisible = False
        )

        #Move agent to center of frame
        self.step(
            action="Teleport",
            position=dict(x=-1.5, y=0.9, z=0),
            rotation=dict(x=0, y=90, z=0),
            horizon=0,
            standing=True,
        )

        #randomly choose 1 type of reward or pick from flag
        rewardTypes = ["Potato", "Tomato", "Apple"]
        rewardType = rewardTypes[args.reward] if args.reward != None else random.sample(rewardTypes, 1)[0]

        #Randomize Materials in the scene
        self.step(
            action="RandomizeMaterials")

        #Randomize Lighting in the scene
        self.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False
        )


        #List of initial poses (receptacle_names' poses)
        initialPoses = []

        #Initialize Object by specifying each object location
        for obj in self.last_event.metadata["objects"]:

            #current Pose of the object
            initialPose = {"objectName": obj["name"],
                            "position": obj["position"],
                            "rotation": obj["rotation"]}

            #Set the Plates location (pre-determined)
            if obj["objectType"] == "Plate" :
                #left plate (z < 0)
                initialPoses.append(
                            {"objectName": obj["name"],
                            "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                            "position": {'x': -0.34, 'y': 1.105, 'z': -0.34}
                            }
                            )

                #right plate (z > 0)
                initialPoses.append(
                            {"objectName": obj["name"],
                            "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                            "position": {'x': -0.34, 'y': 1.105, 'z': 0.34}
                            }
                            )

            #Set the rewards'locations randomly around the plate
            if obj["objectType"] == rewardType:

                #right plate
                for i in range(0,args.left if args.left != None else random.randint(0,8)):
                    initialPoses.append(
                                {"objectName": obj["name"],
                                "rotation": {'x': 0.0, 'y': 0, 'z': 0},
                                "position": {'x': -0.34 + random.uniform(-0.13, 0.13), 'y': 1.15 + 0.001*i, 'z': -0.34 + random.uniform(-0.13, 0.13)}
                                }
                                )
                #left plate
                for i in range(0,args.right if args.right != None else random.randint(0,8)):
                    initialPoses.append(
                                {"objectName": obj["name"],
                                "rotation": {'x': 0.0, 'y': 0, 'z': 0},
                                "position": {'x': -0.34 + random.uniform(-0.13, 0.13), 'y': 1.15 + 0.001*i, 'z': 0.34 + random.uniform(-0.13, 0.13)}
                                }
                                )

            #Ignore reward and receptacles object, they will not be randomized on the table
            if obj["objectType"] in {"Plate", rewardType}:
                pass
            elif not obj["moveable"] and not obj["pickupable"]:
                pass
            else:
                initialPoses.append(initialPose)



        #set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
            action='SetObjectPoses',
            objectPoses = initialPoses,
            placeStationary=False
        )


        #Store all rewards Id in list to be exclude from randomization
        excludedRewardsId = []
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == rewardType:
                excludedRewardsId.append(obj["objectId"]) #useful for randomization of non-rewards

        #randomize all non-rewards objects
        self.step(action="InitialRandomSpawn",
            randomSeed=random.randint(0,10),
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True,
            excludedObjectIds= excludedRewardsId
        )

        #count rewards to get output
        self.out = None
        left = 0
        right = 0

        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == rewardType:
                if obj["position"]["z"] > 0:
                    right += 1
                if obj["position"]["z"] < 0:
                    left +=1
        if left > right:
            self.out = -1
            print("left")
        elif left < right:
            self.out = 1
            print("right")
        else:   #left == right
            self.out = 0
            print("even")

        #dummy move for visual
        self.step("MoveBack", moveMagnitude = 0)
        self.step("MoveBack", moveMagnitude = 0)

    def save_frames_to_file(self):
        from PIL import Image

        image_folder = './'
        print('num frames', len(self.frame_list))
        height, width, channels = self.frame_list[0].shape

        for i, frame in enumerate(tqdm(self.frame_list)):
            img = Image.fromarray(frame)
            img.save("relatives_agent/{}.jpeg".format(i))
        
        print('num frames', len(self.third_party_camera_frames))
        height, width, channels = self.third_party_camera_frames[0].shape

        for i, frame in enumerate(tqdm(self.third_party_camera_frames)):
            img = Image.fromarray(frame)
            img.save("relative_monkey/{}.jpeg".format(i))
        
vid = VideoBenchmark()
# vid.save_frames_to_file()
