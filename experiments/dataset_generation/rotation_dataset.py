from utils.experiment_job import ExperimentJob

expt = ExperimentJob("config/renderer.yaml", ["config/RotationChain_config.yaml"])
expt.run()
