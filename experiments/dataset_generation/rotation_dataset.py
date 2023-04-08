from utils.experiment_job import ExperimentJob

if __name__ == "__main__":
    expt = ExperimentJob(
        "config/linux_renderer.yaml", ["config/RotationChain_config.yaml"]
    )
    expt.run()
