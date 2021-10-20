import os
from PIL import Image
from ai2thor.controller import Controller
from tqdm import tqdm
import yaml
import sys


class Experiment(Controller):
    def __init__(self, controller_args):
        super().__init__(**controller_args)
        self.frame_list = []
        self.saved_frames = []
        self.third_party_camera_frames = []

    def save_frames_to_folder(self, SAVE_DIR, first_person=False):

        if not os.path.isdir(f"frames/{SAVE_DIR}"):
            os.makedirs(f"frames/{SAVE_DIR}")

        fov_frames = self.frame_list if first_person else self.third_party_camera_frames
        print("num frames", len(fov_frames))
        height, width, channels = fov_frames[0].shape

        for i, frame in enumerate(tqdm(fov_frames)):
            img = Image.fromarray(frame)
            img.save(f"frames/{SAVE_DIR}/{i}.jpeg")

    def run(self):
        raise NotImplementedError


def runExperimentJob(renderer_file, experiment_file):
    def str_to_class(classname):
        return getattr(sys.modules[__name__], classname)

    with open(f"{experiment_file}", "r") as stream:
        experiment_data = yaml.safe_load(stream)

    for experiment, parameters in experiment_data.items():
        print(experiment, parameters)
