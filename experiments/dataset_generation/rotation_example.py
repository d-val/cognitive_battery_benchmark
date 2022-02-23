from utils.experiment_tasks.rotation import Rotation

Experiment = Rotation()
Experiment.run()
### UNCOMMENT THE FOLLOWING LINE TO SAVE IMAGES
Experiment.save_frames_to_folder("output/rotation_example")
