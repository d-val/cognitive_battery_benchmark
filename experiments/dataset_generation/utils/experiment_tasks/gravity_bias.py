# -*- coding: utf-8 -*-
import argparse
from ast import parse
import math
from multiprocessing.sharedctypes import Value
import os
import random
import numpy as np
from .utils.experiment import Experiment

# Gravity bias imports
import pickle, yaml, cv2, shutil

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class GravityBias(Experiment):

    def __init__(
        self,
        controller_args={
            "unity_build": "utils/GravityBias.app",
            "width": 300,
            "height": 300,
            "show": True,
        },
        fov=[50, 65],
        seed=0,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.fov = fov
        self.unity_build = controller_args["unity_build"]

        self.controller_args = controller_args

        if not os.path.isdir(self.unity_build):
            raise ValueError("The Gravity Bias build is not in utils. Please download it and re-run this script.")
        self.bin_path = os.path.join(self.unity_build, 'Contents/MacOS/"Gravity Bias"')

        self.outpath = os.path.join(os.getcwd(), ".gravitybiasout/")
        if os.path.isdir(self.outpath):
            # ensures no overlap with previous runs
            shutil.rmtree(self.outpath) 

        os.makedirs(os.path.join(self.outpath + "human_readable/" + "frames/"))
        os.makedirs(os.path.join(self.outpath + "machine_readable"))

    def save_pickle(self, pickle_path, hr_path, frames):
        # Getting 'images'
        frame_names = sorted(frames.keys(), key=lambda x:int(x[:-4].split("_")[-1]))
        images = [np.asarray(frames[img]) for img in frame_names]

        # Getting 'stats'
        with open(os.path.join(hr_path, "experiment_stats.yaml")) as yaml_stream:
            parsed_yaml = yaml.safe_load(yaml_stream)
        stats = parsed_yaml

        # Getting 'label'
        label = parsed_yaml["final_location"]

        # Writing pickle file
        iter_data = {"images": images, "stats": stats, "label":label}
        with open(os.path.join(pickle_path, "iteration_data.pickle"), "wb") as handle:
            pickle.dump(iter_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_video(self, frames, video_path = "experiment_video.mp4"):
        """
        Turns the frames in frames_dir into a video and saves it at the video_path.
        frames: (dict<string, Image>) a mapping from frames' names to their Image instances.
        video_path: (string) the path at which the video is to be saved.
        """

        # Initializing frames and their dims
        frame_names = sorted(frames.keys(), key=lambda x:int(x[:-4].split("_")[-1]))
        init_frame = frames[frame_names[0]]
        height, width, _layers = init_frame.shape  

        # Creating the video at video_path
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 
    
        # Appending the frames to the video one by one
        for image in frame_names: 
            video.write(frames[image]) 
        
        # Deallocating memories taken for window creation
        cv2.destroyAllWindows() 
        video.release()
    
    def get_frames(self, frames_dir):
        """
        Reads the frames in frames_dir and returns a dictionary mapping each frame's name to its Image instance.
        frames_dir: (string) the path of the experiment frames.
        """
        
        images = {}
        for image in os.listdir(frames_dir):
            if image.endswith("png"):
                images[image] = cv2.imread(os.path.join(frames_dir, image))
        return images

    def run(
        self,
        rewardTypes=["Potato", "Tomato", "Apple"],
    ):
        """
        Runs the Gravity Bias build.
        bin_path: (string) the path to the Unity binary build.
        show: (bool) whether to show the experiment.
        record: (bool) whether to record the experiment.
        outdir: (string) path to save recorded frames
        """
        args = ""

        if not self.controller_args["show"]: args += " -batchmode "
        args += f'--outdir "{self.outpath}" -record --seed {self.seed} '
        args += f'--width {self.controller_args["width"]} '
        args += f'--height {self.controller_args["width"]} '
        args += f'--fov {np.random.uniform(self.fov[0], self.fov[1])} '

        # Ensures the experiment app has execution permessions
        os.system(f"chmod +x ./{self.bin_path}")
        os.system(f"xattr -r -d com.apple.quarantine {self.unity_build}")

        # Runs the experiment binary
        os.system(f'./{self.bin_path} {args}')

    def stop(self):
        # Automatically stops
        return

    def save_frames_to_folder(
        self,
        SAVE_DIR,
        first_person=None,
        save_stats=True,
        db_mode=True,
        save_video=True,
    ):
        """
        Redirects the experiment output from the temp path to SAVE_DIR.
        Also can produce a video and experiment stats pickle.
        save_stats: whether to save the experiment stats
        save_video: whether to save the frames as a video
        """
        if os.path.isdir(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)

        try:
            new_path = os.path.abspath(SAVE_DIR)
            shutil.move(self.outpath, new_path)
            self.outpath = new_path
        except OSError:
            raise ValueError(f"The directory {SAVE_DIR} is not empty. Output remain in {self.outpath}")
        except Exception:
            raise ValueError(f"Could not save in {SAVE_DIR}. Output remain in {self.outpath}")
        
        db_SAVE_DIRS = {
                "human": f"{SAVE_DIR}/human_readable",
                "machine": f"{SAVE_DIR}/machine_readable",
            }
        
        if save_video or save_stats:
            frames = self.get_frames(os.path.join(db_SAVE_DIRS["human"], "frames"))
            if save_video:
                self.save_video(frames, os.path.join(SAVE_DIR, "experiment_video.mp4"))
            if save_stats:
                self.save_pickle(db_SAVE_DIRS["machine"], db_SAVE_DIRS["human"], frames)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rotation from file")
    parser.add_argument(
        "-S",
        "--show",
        action="store_true",
        help="show experiment"
    )
    parser.add_argument(
        "saveTo",
        action="store",
        type=str,
        help="which folder to save frames to",
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=0, help="random seed for experiment"
    )
    parser.add_argument(
        "--height", action="store", type=int, default=800, help="height of the frame"
    )
    parser.add_argument(
        "--width", action="store", type=int, default=800, help="width of the frame"
    )
    parser.add_argument(
        "--buildPath", action="store", type=str, default="utils/GravityBias.app", help="path to experiment build"
    )

    args = parser.parse_args()

    experiment = GravityBias(
        {"height": args.height,
         "width": args.width,
         "unity_build": args.buildPath,
         "show": args.show},
        seed=args.seed
    )
    experiment.run(
        case=args.case,
        distances=args.distances,
        rewardTypes=args.rewTypes,
        rewardType=args.rewType,
    )
    experiment.stop()
    experiment.save_frames_to_folder(args.saveTo)
