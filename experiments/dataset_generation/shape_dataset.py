from utils.experiment_job import ExperimentJob

expt = ExperimentJob("config/renderer.yaml", ["config/Shape_config.yaml"])
expt.run()
