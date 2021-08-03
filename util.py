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
    controller.step(
            action="PickupObject",
            objectId=object_ID,
            forceAction=False,
            manualInteract=True
            )
    checkError(controller)

    return controller.last_event

def drop_object(controller):
    """
    controller: current controller

    drop the object holding
    
    return: event after pickup
    """
    controller.step(
            action="DropHandObject",
            forceAction=False
            )
    checkError(controller)
    return controller.last_event

def move_hand(controller, directions):
    """
    controller: current controller

    directions: list of (ahead, right, up) directions
    move the agent's hand according to the directions (ahead, right, up) one at a time

    return: event after the last movement
    """
    for ahead, right, up in directions:
        controller.step(
                action = "MoveHeldObject",
                ahead = ahead,
                right = right,
                up = up,
                forceVisible=False
            )
        checkError(controller)
    return controller.last_event


def move_object(controller, objectId, directions):
    """u
    controller: current controller
    object_ID: unique objectId of the object to be moved
    directions: list of (ahead, right, up) directions to move

    1. pickup the object
    2. move the agent hand according to the directions (ahead, right, up) one at a time
    3. drop the holding object
    return: the event after drop the object
    """
    pickup(controller, objectId)

    move_hand(controller, directions)

    return drop_object(controller)

def checkError(controller):
    """
    print error message if exist
    """
    if controller.last_event.metadata["errorMessage"] != '':
        print(controller.last_event.metadata["errorMessage"])
