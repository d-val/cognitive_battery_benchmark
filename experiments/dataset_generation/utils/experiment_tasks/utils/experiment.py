import logging
import os
import pickle

import imageio
import yaml
from PIL import Image
from ai2thor.controller import Controller

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


class Experiment(Controller):
    def __init__(self, controller_args, fov="front"):
        super().__init__(
            **{
                **{  # local build
                    "local_executable_path": f"{BASE_DIR}/utils/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
                    "agentMode": "default",
                    "scene": "FloorPlan1",
                    # step sizes
                    "gridSize": 0.25,
                    "snapToGrid": False,
                    "rotateStepDegrees": 90,
                    # image modalities
                    "renderDepthImage": False,
                    "renderInstanceSegmentation": False,
                    # camera properties
                    "width": 300,
                    "height": 300,
                    "makeAgentsVisible": False,
                },
                **controller_args,
            }
        )
        self.frame_list = []
        self.saved_frames = []
        self.third_party_camera_frames = []
        self.fov = fov

    def save_frames_to_folder(
        self,
        SAVE_DIR,
        first_person=None,
        save_stats=True,
        db_mode=True,
        save_video=True,
    ):
        fov = first_person if first_person is not None else self.fov
        fov_frames = (
            self.frame_list if self.fov == "front" else self.third_party_camera_frames
        )

        if db_mode:
            db_SAVE_DIRS = {
                "human": f"{SAVE_DIR}/human_readable",
                "machine": f"{SAVE_DIR}/machine_readable",
            }
            for name, folder in db_SAVE_DIRS.items():
                if not os.path.isdir(f"{folder}"):
                    if name == "human":
                        os.makedirs(f"{folder}/frames")
                    else:
                        os.makedirs(f"{folder}")
                with open(
                    f"{folder}/experiment_stats.yaml",
                    "w",
                ) as yaml_file:
                    yaml.dump(self.stats, yaml_file, default_flow_style=False)

                if name == "human":
                    for i, frame in enumerate(fov_frames):
                        img = Image.fromarray(frame)
                        img.save(f"{folder}/frames/frame_{i}.jpeg")
                elif name == "machine":
                    iter_data = {
                        "images": fov_frames,
                        "label": self.label,
                        "stats": self.stats,
                    }
                    with open(f"{folder}/iteration_data.pickle", "wb") as handle:
                        pickle.dump(iter_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if not os.path.isdir(f"{SAVE_DIR}"):
                os.makedirs(f"{SAVE_DIR}")

            elif save_stats:
                with open(
                    f"{SAVE_DIR}/experiment_stats.yaml",
                    "w",
                ) as yaml_file:
                    yaml.dump(self.stats, yaml_file, default_flow_style=False)

        if save_video:
            imageio.mimwrite(
                f"{SAVE_DIR}/experiment_video.mp4",
                fov_frames,
                fps=30,
                quality=9.5,
                macro_block_size=16,
                ffmpeg_log_level="error",
            )

    def label_to_int(self, label):
        if label == "left":
            return -1
        elif label == "right":
            return 1
        else:
            return 0

    def run(self):
        raise NotImplementedError
