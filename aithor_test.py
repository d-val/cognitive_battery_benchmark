# -*- coding: utf-8 -*-
import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
from util import *

#unity directory


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

img_array = []

controller = Controller(

    #local build
    local_executable_path=f"{BASE_DIR}/custom-build.app/Contents/MacOS/AI2-THOR",
    
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
    fieldOfView=90
)


#CONSTANTS
MOVEUP_MAGNITUDE = 0.3
MOVE_RECEP_AHEAD_MAG = 0.3

#receptable z coordinate
pot_zs = []

#receptable name
pots = []

#get_object the z of the rewardId and receptables and also the receptable ids
for obj in controller.last_event.metadata["objects"]:
    if obj["objectType"] == "Egg":
        rewardId = obj["objectId"]
        egg_z = obj["position"]["z"]
    if obj["objectType"] == "Pot":
        pots.append(obj["name"])
        pot_zs.append(obj["position"]["z"])

#sample 1 random receptable to put the rewardId in
correct_pot_z = random.sample(pot_zs,1)[0]
egg_move_left_mag = correct_pot_z - egg_z

#Move agent to fit the screen
controller.step("MoveRight")
img_array.append(controller.last_event.frame)

#move the reward to the pre-selected receptable then drop it
move_object(controller, rewardId, [(0,0, MOVEUP_MAGNITUDE), (0, -egg_move_left_mag, 0)])
img_array.append(controller.last_event.frame)

#Swap 2 receptables
def swap(swap_receptables):
  """ swap_receptables: list of 2 pots object to swap
  return None
  """
  event = controller.last_event
  recep1_name = swap_receptables[0]
  recep2_name = swap_receptables[1]
  recep1_id = get_objectId(recep1_name, controller)
  recep2_id = get_objectId(recep2_name, controller)

  #calculate the z-different to move the receps
  z_different = get_object(recep1_name, controller)["position"]["z"] - get_object(recep2_name, controller)["position"]["z"]

  #move first recep far away
  move_object(controller, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (MOVE_RECEP_AHEAD_MAG, 0, 0)])
  img_array.append(controller.last_event.frame)

  #move 2nd recep to 1st recep place
  move_object(controller, recep2_id, [(0, 0, MOVEUP_MAGNITUDE), (0, -z_different, 0)])
  img_array.append(controller.last_event.frame)

  # every time an object is moved, its id is changed
  # update 1st receptable ID
  recep1_id = get_objectId(recep1_name, controller)

  #move 1st recep to second recep place
  move_object(controller, recep1_id, [(0, 0, MOVEUP_MAGNITUDE), (0, z_different, 0), (-MOVE_RECEP_AHEAD_MAG, 0, 0)])
  
  img_array.append(controller.last_event.frame)

     
for i in range(10):
    swap(random.sample(pots,2))


#dummy moves to visualize
controller.step("MoveBack") 



# if os.path.exists('project.mp4'):
#     os.remove('project.mp4')
# out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'FMP4'), 1, (2000,2000))
# for frame in img_array:
#     out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# cv2.destroyAllWindows()
# out.release()



