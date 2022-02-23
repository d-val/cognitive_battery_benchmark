from utils.experiment_tasks.simple_swap import SimpleSwap

SimpleSwapExperiment = SimpleSwap(seed=100)
SimpleSwapExperiment.run()
### UNCOMMENT THE FOLLOWING LINE TO SAVE IMAGES
SimpleSwapExperiment.save_frames_to_folder("output/simple_swap_example")
