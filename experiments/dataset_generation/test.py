from experiments.dataset_generation.simple_swap import SimpleSwap

SimpleSwapExperiment = SimpleSwap()
SimpleSwapExperiment.run()
SimpleSwapExperiment.save_frames_to_folder("output")
