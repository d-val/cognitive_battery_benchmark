def get_object(receptable_name, controller):
    """
    receptable_name: unique name of the receptable object
    controller: current controller

    return: object in metadata of controller if exitst, else None
    """
    for obj in controller.last_event.metadata["objects"]:
        if obj["name"] == receptable_name:
            return obj
    return None


def get_objectId(receptable_name, controller):
    """
    receptable_name: unique name of the receptable object
    controller: current controller

    return: objectId in metadata of controller if exitst, else None
    """
    return get_object(receptable_name, controller)["objectId"]


def pickup(controller, object_ID):
    """
    controller: current controller
    object_ID: unique objectId of the object to be picked up

    pickup the object

    return: event after pickup
    """

    # controller.step("PausePhysicsAutoSim")
    controller.step(
        action="PickupObject",
        objectId=object_ID,
        forceAction=False,
        manualInteract=True,
        # manualInteract=False,
    )
    # images = []
    # # for i in range(10):
    # while not controller.last_event.metadata["isSceneAtRest"]:
    #     controller.step(
    #         action="AdvancePhysicsStep",
    #         timeStep=0.01
    #     )
    #     last_image = controller.last_event.frame
    #     images.append(last_image)

    # controller.step("UnpausePhysicsAutoSim")
    checkError(controller)

    return controller.last_event


def drop_object(controller, frame_list, third_party_camera_frames):
    """
    controller: current controller

    drop the object holding

    return: event after pickup
    """

    # TODO make these changes after bug is fixed
    # images = []
    # controller.step("PausePhysicsAutoSim")
    controller.step(action="DropHeldObject", forceAction=False)
    # while not controller.last_event.metadata["isSceneAtRest"]:
    #     controller.step(
    #         action="AdvancePhysicsStep",
    #         timeStep=0.01
    #     )
    #     frame_list.append(last_image)

    # controller.step("UnpausePhysicsAutoSim")
    checkError(controller)
    return controller.last_event, frame_list, third_party_camera_frames


def move_hand(controller, directions, frame_list, third_party_camera_frames):
    """
    controller: current controller

    directions: list of (ahead, right, up) directions
    move the agent's hand according to the directions (ahead, right, up) one at a time

    return: event after the last movement
    """

    # images = []
    for ahead, right, up in directions:
        # for i in range(10):
        # while not controller.last_event.metadata["isSceneAtRest"]:
        #     controller.step(
        #         action="AdvancePhysicsStep",
        #         timeStep=0.01
        #     )
        #     # images.append(last_image)

        controller.step(
            action="MoveHeldObject", ahead=ahead, right=right, up=up, forceVisible=False
        )
        last_image = controller.last_event.frame
        frame_list.append(last_image)
        third_party_camera_frames.append(
            controller.last_event.third_party_camera_frames[0]
        )
        checkError(controller)
    return controller.last_event, frame_list, third_party_camera_frames


def move_object(
    controller, objectId, directions, frame_list, third_party_camera_frames
):
    """u
    controller: current controller
    object_ID: unique objectId of the object to be moved
    directions: list of (ahead, right, up) directions to move

    1. pickup the object
    2. move the agent hand according to the directions (ahead, right, up) one at a time
    3. drop the holding object
    return: the event after drop the object
    """
    last_event = pickup(controller, objectId)
    frame_list.append(last_event.frame)
    third_party_camera_frames.append(last_event.third_party_camera_frames[0])
    directions = interpolate_between_2points(directions, num_interpolations=10)
    _, frame_list, third_party_camera_frames = move_hand(
        controller, directions, frame_list, third_party_camera_frames
    )

    last_event, framelist, third_party_camera_frames = drop_object(
        controller, frame_list, third_party_camera_frames
    )
    return last_event, framelist, third_party_camera_frames


def checkError(controller):
    """
    print error message if exist
    """
    # if controller.last_event.metadata["errorMessage"] != '':
    #     print(controller.last_event.metadata["errorMessage"])


def interpolate_between_2points(directions, num_interpolations):
    all_directions = []
    for direction in directions:
        a = direction[0] / num_interpolations
        b = direction[1] / num_interpolations
        c = direction[2] / num_interpolations
        for i in range(num_interpolations):
            all_directions.append((a, b, c))

    return all_directions
