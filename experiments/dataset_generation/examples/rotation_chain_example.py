from utils.experiment_tasks.rotationchain import RotationChain

Experiment = RotationChain()
Experiment.run()
### UNCOMMENT THE FOLLOWING LINE TO SAVE IMAGES
Experiment.save_frames_to_folder("output/rotation_chain_example")
