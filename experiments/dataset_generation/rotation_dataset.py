from utils.experiment_job import ExperimentJob

expt = ExperimentJob("config/linux_renderer.yaml", ["config/RotationChain_config.yaml"])
expt.run()
