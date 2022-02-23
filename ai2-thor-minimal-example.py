from ai2thor.controller import Controller

controller = Controller(scene="FloorPlan10")
event = controller.step(action="RotateRight")
metadata = event.metadata
print("success!", event, event.metadata.keys())
