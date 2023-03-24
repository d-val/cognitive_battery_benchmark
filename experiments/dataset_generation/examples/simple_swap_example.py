from utils.experiment_tasks.simple_swap import SimpleSwap

SimpleSwapExperiment = SimpleSwap(seed=100)
SimpleSwapExperiment.run()
### THE FOLLOWING LINE SAVES IMAGES OF THE SIMULATION
SimpleSwapExperiment.save_frames_to_folder("output/simple_swap_example")
