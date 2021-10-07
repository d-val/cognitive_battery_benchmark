# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from ai2thor.controller import Controller
import random
import cv2
import random
from tqdm import tqdm
import math

# unity directory
from experiment import Experiment
from utils.util import move_object

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Rotation(Experiment):

    def __init__(self, case=1, seed=0):
        # TODO: ask about the case here

        # set case according to the spreadsheet
        self.frame_list = []
        self.saved_frames = []
        self.third_party_camera_frames = []
        super().__init__(
            {
                # local build
                "local_executable_path": f"{BASE_DIR}/utils/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
                "agentMode": "default",
                "visibilityDistance": 5,
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
                "makeAgentsVisible": False
            },
            seed
        )

        self.step(
            action="AddThirdPartyCamera",
            position=dict(x=1.5, y=1.8, z=0),
            rotation=dict(x=0, y=270, z=0),
            fieldOfView=90
        )

        # Randomize Materials in the scene
        self.step(
            action="RandomizeMaterials")

        # Randomize Lighting in the scene
        self.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False
        )

        rewardTypes = ["Potato", "Tomato", "Apple"]

        self.rewardType = random.sample(rewardTypes, 1)[0]

        # List of initial poses (receptacle_names' poses)
        initialPoses = []
        # A list of receptacle object types to exclude from valid receptacles that can be randomly chosen as a spawn location.
        # https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#random-spawn-excludedreceptacles

        excludeList = []  # Egg and Pot exclude from randomization
        randomObjects = []  # store all other Pickupable objects

        # set distance of cups to the center of tray
        d1 = -0.4
        d2 = 0
        d3 = 0.4

        # set number of rotation, 11 for 360 degree and 6 for 180 degree
        # 11 means rotate 10 times 36 degree each and 6 means rotate 5 times 
        num_rotate = 0

        # rotate 180, food in middle
        if case == 1:
            num_rotate = 6
            food_dist = d2
        # rotate 360, food in left or right
        if case == 2:
            num_rotate = 11
            food_dist = random.choice([d1, d3])
        # rotate 180, food in left or right
        if case == 3:
            num_rotate = 6
            food_dist = random.choice([d1, d3])
        # Initialize Object by specifying each object location, receptacle and rewward are set to pre-determined locations, the remaining stays at the same place
        # and will be location randomized later
        for i in range(0, num_rotate):
            # empty initial poses after each iteration to avoid duplicate
            initialPoses = []
            for obj in self.last_event.metadata["objects"]:
                angle = i * 36
                angle_radian = 2 * math.pi * angle / 360
                # current Pose of the object
                initialPose = {"objectName": obj["name"],
                               "position": obj["position"],
                               "rotation": obj["rotation"]}

                # if obj["name"] == "Tray":
                #     #mid occluder
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': angle, 'z': 0},
                #                 "position": {'x': 0, 'y': 1.105, 'z': 0}
                #                 }
                #                 )
                # if obj["objectType"] == "Potato":
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                #                 "position": {'x': 0 + 0.4 * math.sin(angle_radian), 'y': 1.205, 'z': 0.4*math.cos(angle_radian)}
                #                 }
                #                 )
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                #                 "position": {'x': 0 + 0 * math.sin(angle_radian), 'y': 1.205, 'z': 0*math.cos(angle_radian)}
                #                 }
                #                 )
                #     initialPoses.append(
                #                 {"objectName": obj["name"],
                #                 "rotation": {'x': -0.0, 'y': 0, 'z': 0},
                #                 "position": {'x': 0 - 0.4 * math.sin(angle_radian), 'y': 1.205, 'z': -0.4*math.cos(angle_radian)}
                #                 }
                #                 )
                if obj["name"] != "Tray" and obj["objectType"] != "Potato" and obj["name"][:4] != "Cup1":
                    initialPoses.append(initialPose)

            initialPoses.append(
                {"objectName": "Tray",
                 "rotation": {'x': -0.0, 'y': angle, 'z': 0},
                 "position": {'x': 0, 'y': 1.105, 'z': 0}
                 }
            )
            initialPoses.append(
                {"objectName": "Cup1",
                 "rotation": {'x': -0.0, 'y': angle, 'z': 180},
                 "position": {'x': 0 + d3 * math.sin(angle_radian), 'y': 1.505, 'z': d3 * math.cos(angle_radian)}
                 }
            )
            initialPoses.append(
                {"objectName": "Cup1",
                 "rotation": {'x': -0.0, 'y': angle, 'z': 180},
                 "position": {'x': d2 * math.sin(angle_radian), 'y': 1.505, 'z': d2 * math.cos(angle_radian)}
                 }
            )
            initialPoses.append(
                {"objectName": "Cup1",
                 "rotation": {'x': -0.0, 'y': angle, 'z': 180},
                 "position": {'x': d1 * math.sin(angle_radian), 'y': 1.505, 'z': d1 * math.cos(angle_radian)}
                 }
            )
            initialPoses.append(
                {"objectName": "Potato_35885ea7",
                 "rotation": {'x': -0.0, 'y': angle, 'z': 0},
                 "position": {'x': food_dist * math.sin(angle_radian), 'y': 1.205,
                              'z': food_dist * math.cos(angle_radian)}
                 }
            )
            print(len(initialPoses))
            # set inital Poses of all objects, random objects stay in the same place, chosen receptacle spawn 3 times horizontally on the table
            self.step(
                action='SetObjectPoses',
                objectPoses=initialPoses,
                placeStationary=False
            )

            # add frame to corresponding frame list
            self.frame_list.append(self.last_event.frame)
            self.third_party_camera_frames.append(self.last_event.third_party_camera_frames[0])
            # initial state, lift up cup to show food
            if i == 0:
                for obj in self.last_event.metadata["objects"]:
                    if obj["name"][:4] == "Cup1":
                        move_object(self, obj["objectId"], [(0, 0, 0.4)], self.frame_list,
                                    self.third_party_camera_frames)
        out = None
        # return value
        # 1 = right, 0 = middle, -1 = left
        for obj in self.last_event.metadata["objects"]:
            if obj["name"] == "Potato_35885ea7":
                dist = obj["position"]["z"]
                if dist > 0.3:
                    out = 1
                if dist < -0.3:
                    out = -1
                if abs(dist) < 0.1:
                    out = 0
        print(out)

        # dummy moves for debug
        self.step("MoveBack")
        self.step("MoveAhead")

    def save_frames_to_file(self, SAVE_DIR):

        if not os.path.isdir(SAVE_DIR):
            os.makedirs(f"{SAVE_DIR}/rotation_agent_view")
            os.makedirs(f"{SAVE_DIR}/rotation_monkey_view")

        print('num frames', len(self.frame_list))
        height, width, channels = self.frame_list[0].shape

        for i, frame in enumerate(tqdm(self.frame_list)):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/rotation_agent_view/{i}.jpeg")

        print('num frames', len(self.third_party_camera_frames))
        height, width, channels = self.third_party_camera_frames[0].shape

        for i, frame in enumerate(tqdm(self.third_party_camera_frames)):
            img = Image.fromarray(frame)
            img.save(f"{SAVE_DIR}/rotation_monkey_view/{i}.jpeg")
