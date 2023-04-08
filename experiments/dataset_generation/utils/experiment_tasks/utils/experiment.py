import logging
import os
import pickle
from sys import platform
import imageio
import yaml
from PIL import Image
from ai2thor.controller import Controller
from skimage import img_as_ubyte
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def list_to_folder(folder, frame_list, npy=False):
    for i, frame in enumerate(frame_list):
        if npy:
            np.save(f"{folder}/frame_{i}.npy", frame)
        else:
            img = Image.fromarray(frame)
            img.save(f"{folder}/frame_{i}.png")


def save_frames_to_video(save_dir, frame_list, video_name):

    imageio.mimwrite(
        f"{save_dir}/{video_name}.mp4",
        [np.uint8(frame) for frame in frame_list],
        fps=30,
        quality=9.5,
        macro_block_size=16,
        ffmpeg_log_level="error",
    )
    return


class Experiment(Controller):
    def __init__(self, controller_args, fov="front"):
        config_dict = {  # local build
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
        }
        config = {
            **config_dict,
            **controller_args,
        }
        super().__init__(**config)
        self.frame_list = []
        if config.get("renderDepthImage", False):
            self.depth_list = []
        else:
            self.depth_list = None
        if config.get("renderInstanceSegmentation", False):
            self.segmentation_list = []
        else:
            self.segmentation_list = None
        self.third_party_camera_frames = []
        self.fov = fov

    def save_frames_to_folder(
        self,
        save_dir,
        first_person=None,
        save_stats=True,
        db_mode=True,
        save_video=True,
        save_raw_data=False,
    ):
        fov_frames = (
            self.frame_list if self.fov == "front" else self.third_party_camera_frames
        )

        if db_mode:
            db_SAVE_DIRS = {
                "human": f"{save_dir}/human_readable",
                "machine": f"{save_dir}/machine_readable",
            }
            for name, folder in db_SAVE_DIRS.items():
                if not os.path.isdir(f"{folder}"):
                    if name == "human":
                        os.makedirs(f"{folder}/frames")
                    else:
                        os.makedirs(f"{folder}")
                        if self.depth_list is not None:
                            os.makedirs(f"{folder}/depth_frames")
                        if self.segmentation_list is not None:
                            os.makedirs(f"{folder}/segmentation_frames")
                with open(
                    f"{folder}/experiment_stats.yaml",
                    "w",
                ) as yaml_file:
                    yaml.dump(self.stats, yaml_file, default_flow_style=False)

                if name == "human":
                    list_to_folder(f"{folder}/frames", fov_frames)

                elif name == "machine":
                    iter_data = {
                        "images": fov_frames,
                        "depth": self.depth_list,
                        "segmentation": self.segmentation_list,
                        "label": self.label,
                        "stats": self.stats,
                    }
                    if len(self.depth_list) > 0:
                        list_to_folder(
                            f"{folder}/depth_frames", self.depth_list, npy=True
                        )

                    if len(self.segmentation_list) > 0:
                        list_to_folder(
                            f"{folder}/segmentation_frames",
                            self.segmentation_list,
                            npy=True,
                        )

                    if save_raw_data:
                        with open(f"{folder}/iteration_data.pickle", "wb") as handle:
                            pickle.dump(
                                iter_data, handle, protocol=pickle.HIGHEST_PROTOCOL
                            )
        else:
            if not os.path.isdir(f"{save_dir}"):
                os.makedirs(f"{save_dir}")
            elif save_stats:
                with open(
                    f"{save_dir}/experiment_stats.yaml",
                    "w",
                ) as yaml_file:
                    yaml.dump(self.stats, yaml_file, default_flow_style=False)

        if save_video:
            save_frames_to_video(save_dir, fov_frames, "experiment_video")

    def update_frames(self):
        self.frame_list.append(self.last_event.frame)
        if self.depth_list is not None:
            self.depth_list.append(self.last_event.depth_frame)
        if self.segmentation_list is not None:
            self.segmentation_list.append(self.last_event.instance_segmentation_frame)

    def run(self):
        raise NotImplementedError
