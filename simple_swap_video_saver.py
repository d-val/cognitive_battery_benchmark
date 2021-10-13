# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
from util import *
import random
from tqdm import tqdm
from math import erf, sqrt
#unity directory
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

#PREDETERMINED CONSTANTS
MOVEUP_MAGNITUDE = 0.45
MOVE_RECEP_AHEAD_MAG = 0.45

class VideoBenchmark(Controller):

    def __init__(self):
        import argparse

        msg = "Adding description"
        # Initialize parser
        my_parser = argparse.ArgumentParser(description = msg)

        my_parser.add_argument('--height', action='store', type=int, help = "height of the frame")
        my_parser.add_argument('--width', action='store', type=int, help = "width of the frame")
        my_parser.add_argument('-r', '--reward', action='store', type=int, help = "reward type \n Egg = 0\n Potato = 1\n Apple = 2\n Tomato = 3")
        my_parser.add_argument('-c', '--container', action='store', type=int, help = "receptacle type \n Pot = 0\n Mug = 1\n Cup = 2")
        my_parser.add_argument('--case', action='store', type=int, help = "case number \n Single Transposition = 1\n Double unbaited transposition = 2\n Double baited transposition = 3")
        my_parser.add_argument('--position', action='store', type=int, help = "reward initial position \n right = 0\n middle = 1\n left = 2")

        args = my_parser.parse_args()

        # random.seed(10)

        #initialize lists to store frame from agent view (frame_list) and monkey view (third_party_camera_frames)
        self.frame_list = []
        self.third_party_camera_frames = []

        #expected answer to feed ML model, 0 = right, 1 = middle, 2 = left
        self.out = None
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
            fieldOfView=random.randint(90,140)
        )

        #add camera that captures agent action from monkey's perspective
        self.step(
            action="AddThirdPartyCamera",
            position=dict(x=-1.5, y=1, z=0),
            rotation=dict(x=0, y=90, z=0),
            fieldOfView=90
        )

        #Randomize Materials in the scene
        self.step(
            action="RandomizeMaterials")

        #Randomize Lightings in the scene
        self.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False
        )

        #Possible receptacle types
        receptacleTypes = ["Pot", "Mug", "Cup"]

        #Set Receptable type from paste in argument or randomly choose one
        receptacleType = receptacleTypes[args.container] if args.container != None else random.sample(receptacleTypes, 1)[0]

        #Possible reward objects (Egg, Ball, ...) types
        self.rewardTypes = ["Egg", "Potato", "Tomato", "Apple"]

        #Set reward type from paste in argument or randomly choose one
        self.rewardType = self.rewardTypes[args.reward] if args.reward != None else random.sample(self.rewardTypes, 1)[0]
        
        print("Receptacle: ", receptacleType)
        print("Reward: ", self.rewardType)

        #List of initial poses (receptacle_name_and_z_coor' poses)
        initialPoses = []
        #A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
        #https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

        excludeList = []                #Egg and Pot exclude from randomization
        randomObjects = []              #store all other Pickupable objects

        #Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
        #and will be location randomized later
        for obj in self.last_event.metadata["objects"]:
            #current Pose of the object
            initialPose = {"objectName": obj["name"],
                              "position": obj["position"],
                              "rotation": obj["rotation"]}

            #Set reward inital position (pre-determined) to the right of the table
            if obj["objectType"] == self.rewardType:
                initialPoses.append(
                            {"objectName": obj["name"],
                            "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                            "position": {'x': -0.4300207197666168, 'y': 1.126484751701355, 'z': -0.9255303740501404}
                            }
                            )

            #Set recetacles location, initialize 3 times on the table at pre-determined positions
            if obj["objectType"] == receptacleType:
                initialPoses.append(
                            {"objectName": obj["name"],
                            "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                            "position": {'x': -0.4351297914981842, 'y': 1.1031372547149658, 'z': 0.7}
                            }
                            )
                initialPoses.append({"objectName": obj["name"],
                            "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                            "position": {'x': -0.4351317286491394, 'y': 1.1031371355056763, 'z': -7.855288276914507e-05}
                            }
                            )
                initialPoses.append({"objectName": obj["name"],
                            "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                            "position": {'x': -0.4351297914981842, 'y': 1.1031371355056763, 'z': -0.7}
                            }
                            )
            #Ignore reward and receptacles object, they will not be randomized place behind the table
            if obj["objectType"] in [self.rewardType] + receptacleTypes:
                pass
            else:
                initialPoses.append(initialPose)



        #set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
        self.step(
          action='SetObjectPoses',
          objectPoses = initialPoses
        )

        #exclude the chosen reward and receptacles from location randomization,
        # only randomize pickupable objects
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] in [self.rewardType] + receptacleTypes:
                excludeList.append(obj["objectId"])
            elif obj["pickupable"]:
                randomObjects.append(obj["objectId"])


        #exclude all but 1 random objects to show randomly on the table
        excludeRandomObjects = random.sample(randomObjects, len(randomObjects)-1)
        excludeRandomObjects = []
        #https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-randomseed
        #InitialRandomSpawn attempts to randomize the position of Pickupable objects, placing them in any valid receptacle they could be placed in within the scene.
        self.step(action="InitialRandomSpawn",
            randomSeed=random.randint(0,10),
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True,
            numDuplicatesOfType = [
            ],

            #Objects could randomly spawn in any suitable receptacles except for the simulating receptacles themselves
            excludedReceptacles= [receptacleType],
            excludedObjectIds= excludeList + excludeRandomObjects
        )

        #receptacle z coordinate to move reward in
        # receptacle_z = []

        #receptacle name and z coor
        self.receptacle_name_and_z_coor = []

        #get the z coordinates of the rewardId (Egg) and receptacles (Pot) and also get the receptacle ids
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                rewardId = obj["objectId"]
                reward_z = obj["position"]["z"]
            if obj["objectType"] == receptacleType:
                self.receptacle_name_and_z_coor.append((obj["name"], obj["position"]["z"]))

        #sort receptacle by z coordinate from positive to negative
        self.receptacle_name_and_z_coor.sort(key = lambda x : -x[1])

        #sample 1 random receptacle to put the reward (Egg) in or paste in from argument
        chosen_receptacle_z = self.receptacle_name_and_z_coor[args.position][1] if args.position != None else random.sample(self.receptacle_name_and_z_coor,1)[0][1]

        #Calculate how much the egg should be moved to the left to be on top of the intended Pot
        reward_move_left_mag = chosen_receptacle_z - reward_z

        #Move agent to fit the screen
        self.step("MoveRight")

        #move the reward to the pre-selected receptacle then drop it
        _, self.frame_list, self.third_party_camera_frames = move_object(self, rewardId, [(0,0, MOVEUP_MAGNITUDE), (0, -reward_move_left_mag, 0), (0, 0, -MOVEUP_MAGNITUDE)], self.frame_list, self.third_party_camera_frames)
        # self.frame_list.append(self.last_event.frame)

    #Swap 2 receptacles
    def swap(self, swap_receptacles):
      """ swap_receptacles: list of 2 receptacle_name_and_z_coor object to swap
      return None
      """
      event = self.last_event
      recep1_name = swap_receptacles[0][0]
      recep2_name = swap_receptacles[1][0]
      recep1_id = get_objectId(recep1_name, self)
      recep2_id = get_objectId(recep2_name, self)

      #calculate the z-different to move the receps
      z_different = get_object(recep1_name, self)["position"]["z"] - get_object(recep2_name, self)["position"]["z"]

      #move first recep far away
      # move_object(self, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)])
      _, self.frame_list, self.third_party_camera_frames = move_object(self, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)], self.frame_list, self.third_party_camera_frames)
      # self.play(move_object(self, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)], self.frame_list))
      self.frame_list.append(self.last_event.frame)
      self.third_party_camera_frames.append(self.last_event.third_party_camera_frames[0])
      #move 2nd recep to 1st recep place
      _, self.frame_list, self.third_party_camera_frames = move_object(self, recep2_id, [(0, 0, MOVEUP_MAGNITUDE), (0, -z_different, 0)], self.frame_list, self.third_party_camera_frames)
      # self.frame_list.append(self.last_event.frame)

      # every time an object is moved, its id is changed
      # update 1st receptacle ID
      recep1_id = get_objectId(recep1_name, self)

      #move 1st recep to second recep place
      _, self.frame_list, self.third_party_camera_frames = move_object(self, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (0, z_different, 0), (-MOVE_RECEP_AHEAD_MAG, 0, 0)], self.frame_list, self.third_party_camera_frames)

      # self.frame_list.append(self.last_event.frame)

    def perform_action(self):
        self.swap(random.sample(self.receptacle_name_and_z_coor,2))

        #get reward final z coordinates
        for obj in self.last_event.metadata["objects"]:
            if obj["objectType"] == self.rewardType:
                reward_final_z = obj["position"]["z"]

        out = None

        #determine which receptacle that reward finally in from monkey perspective
        #0 = right, 1 = middle, 2 = left
        if reward_final_z > -1 and reward_final_z < -0.35:
            out = 2
            print("left")
        elif reward_final_z >= -0.35 and reward_final_z <= 0.35:
            out = 1
            print("middle")
        elif reward_final_z > 0.35 and reward_final_z < 1:
            out = 0
            print("right")
        #dummy moves for debugging purposes
        self.out = out
        self.step("MoveBack", moveMagnitude = 0)
        self.step("MoveBack", moveMagnitude = 0)

        


    def save_frames_to_file(self):
        from PIL import Image

        image_folder = './'
        print('num frames', len(self.frame_list))
        height, width, channels = self.frame_list[0].shape

        for i, frame in enumerate(tqdm(self.frame_list)):
            img = Image.fromarray(frame)
            img.save("frames/{}.jpeg".format(i))
        
        print('num frames', len(self.third_party_camera_frames))
        height, width, channels = self.third_party_camera_frames[0].shape

        for i, frame in enumerate(tqdm(self.third_party_camera_frames)):
            img = Image.fromarray(frame)
            img.save("videos/{}.jpeg".format(i))
        

vid = VideoBenchmark()
vid.perform_action()
# vid.save_frames_to_file()





