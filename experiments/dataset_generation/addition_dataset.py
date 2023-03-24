from utils.experiment_job import ExperimentJob

expt = ExperimentJob("config/renderer.yaml", ["config/Addition_config.yaml"])
expt.run()
