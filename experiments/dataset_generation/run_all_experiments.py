from utils.experiment_job import ExperimentJob

expt = ExperimentJob("config/renderer.yaml", ["config/AllExperiments_config.yaml"])
expt.run()
