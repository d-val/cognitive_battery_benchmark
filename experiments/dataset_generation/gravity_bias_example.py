from utils.experiment_tasks.gravity_bias import GravityBias

Experiment = GravityBias()
Experiment.run(play_speed=7)
# ### UNCOMMENT THE FOLLOWING LINE TO SAVE IMAGES
Experiment.save_frames_to_folder("output/gravity_bias_example")
