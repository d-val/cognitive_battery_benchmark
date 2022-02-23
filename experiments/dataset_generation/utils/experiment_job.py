import argparse
import datetime
import os
import re
import sys
import yaml

from tqdm import tqdm

from .experiment_tasks.addition_numbers import AdditionNumbers
from .experiment_tasks.object_permanence import ObjectPermanence
from .experiment_tasks.relative_numbers import RelativeNumbers
from .experiment_tasks.rotation import Rotation
from .experiment_tasks.shape import Shape
from .experiment_tasks.simple_swap import SimpleSwap
from .experiment_tasks.gravity_bias import GravityBias


class ExperimentJob:
    def __init__(
        self, renderer_file, experiment_files: list, test_init=False, test_run=False
    ):
        self.experiments = {}

        with open(f"{renderer_file}", "r") as stream:
            self.renderer_data = yaml.safe_load(stream)

        self.experiment_data = {}
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

    def run(self, name=None, folder_name="output", seed_pattern="iterative"):
        self.jobName = (
            re.sub(r"[^\w\d-]", "_", str(datetime.datetime.now()))
            if name is None
            else name
        )
        self.make_folder(f"{folder_name}/{self.jobName}")

        with open(f"{folder_name}/{self.jobName}/renderer.yaml", "w") as yaml_file:
            yaml.dump(self.renderer_data, yaml_file, default_flow_style=False)

        with open(f"{folder_name}/{self.jobName}/experiments.yaml", "w") as yaml_file:
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
                    f"{folder_name}/{self.jobName}/{experiment}/{iteration}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SimpleSwap from file")
    parser.add_argument(
        "renderer_file",
        action="store",
        type=str,
        help="location of renderer config",
    )
    parser.add_argument(
        "experiment_files",
        action="store",
        type=str,
        help="location of experiment file configs",
    )
    parser.add_argument(
        "--test_init",
        action="store",
        type=bool,
        default=False,
        help="whether to test init before run",
    )
    parser.add_argument(
        "--test_run",
        action="store",
        type=bool,
        default=False,
        help="whether to test run before running all",
    )
    parser.add_argument(
        "--jobName",
        action="store",
        type=str,
        default=None,
        help="name of job",
    )
    parser.add_argument(
        "--seedPattern",
        action="store",
        type=str,
        default="iterative",
        help="pattern of seeds for iterations of runs",
    )
    args = parser.parse_args()

    job = ExperimentJob(
        renderer_file=args.renderer_file,
        experiment_files=args.experiment_files.split(),
        test_init=args.test_init,
        test_run=args.test_run,
    )

    job.run(name=args.jobName, seed_pattern=args.seedPattern)

# TODO: look into argparse 'choices'
