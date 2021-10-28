import os, sys, yaml, datetime
from shutil import copyfile
from experiments.addition_numbers import AdditionNumbers
from experiments.relative_numbers import RelativeNumbers
from experiments.rotation import Rotation
from experiments.simple_swap import SimpleSwap
from tqdm import tqdm


class ExperimentJob:
    def __init__(self, renderer_file, experiment_files, test_init=False, test_run=False):
        self.experiments = {}

        with open(f"{renderer_file}", "r") as stream:
            self.renderer_data = yaml.safe_load(stream)

        for experiment_file in experiment_files:
            with open(f"{experiment_file}", "r") as stream:
                self.experiment_data.update(yaml.safe_load(stream))

        for experiment, parameters in self.experiment_data.items():
            if not all(
                x in ["init", "run", "iterations", "controllerArgs"] for x in parameters
            ):
                raise AssertionError(
                    f"Unknown field found for {experiment} in YAML file."
                )
            experimentClass = self.str_to_class(experiment)

            if test_init or test_run:
                experimentClass = experimentClass(
                    **{**parameters["init"], **self.renderer_data}
                )
                if test_run:
                    experimentClass.run(**parameters["run"])

    def run(self, name=None, seed_pattern="iterative"):
        self.jobName = datetime.datetime.now() if name is None else name
        self.make_folder(f"frames/{self.jobName}")

        with open(f"frames/{self.jobName}/renderer.yaml", "w") as yaml_file:
            yaml.dump(self.renderer_data, yaml_file, default_flow_style=False)

        with open(f"frames/{self.jobName}/experiments.yaml", "w") as yaml_file:
            yaml.dump(self.experiment_data, yaml_file, default_flow_style=False)

        for experiment, parameters in self.experiment_data.items():
            print(
                f'Running Experiment: {experiment} | {parameters["iterations"]} Iterations'
            )
            for iteration in tqdm(range(parameters["iterations"])):
                if seed_pattern == "iterative":
                    seed = iteration
                experimentClass = self.str_to_class(experiment)(
                    {**self.renderer_data, **parameters.get("controllerArgs", {})},
                    **parameters.get("init", {}),
                    seed=seed,
                )
                experimentClass.run(**parameters["run"])
                experimentClass.stop()
                experimentClass.save_frames_to_folder(
                    f"frames/{self.jobName}/{experiment}/{iteration}"
                )

    @staticmethod
    def make_folder(name):
        if not os.path.isdir(name):
            os.makedirs(name)
        else:
            raise Exception("Job folder already exists.")

    @staticmethod
    def str_to_class(classname):
        return getattr(sys.modules[__name__], classname)
