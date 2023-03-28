from utils.experiment_job import ExperimentJob

expt = ExperimentJob("config/linux_renderer.yaml", ["config/RelativeNumbers_config.yaml"])
expt.run()
