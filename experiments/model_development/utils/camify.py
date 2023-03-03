"""camify.py: Generates class activation mapping (CAM) visualizations"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import os

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def normalize(arr):
    """
    Normalizes an array to be in the [0, 1].

    :param np.ndarray arr: an array to normalize
    :return: a copy of arr normalized to be in the [0, 1] range.
    :rtype: np.ndarray
    """
    return (arr - arr.min()) / (arr.max() - arr.min())


def camify(dataset, model, indices, cam_fn=GradCAM, outpath="gifs", verbose=True):
    """
    Generates and saves CAM visualizations of a given model on given elements of a dataset.

    :param FramesDataset dataset: a dataset of generated videos.
    :param CNNLSTM model: a model
    :param list indices: the indices of videos to create visualizations of in dataset
    :param type cam_fn: a type of CAM visualization technique (e.g. GradCAM, EigenCAM)
    :param str outpath: the path to save the visualizations at.
    :param bool verbose: whether or not to show progress in stdout.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    model.to(device)
    target_layers = [model.cnn.layer4[-1]]

    for idx in tqdm(indices) if verbose else indices:
        fig, axs = plt.subplots(1, 2, figsize=(14, 8.75))
        (ax_orig, ax_vis) = axs.flatten()

        # Getting and saving original video
        video_orig, class_idx = dataset[idx]
        im_orig = ax_orig.imshow(normalize(video_orig[0, :, :, ::-1]))

        # Getting CAM visualization
        input_tensor = np.asarray(video_orig, dtype=np.float32).transpose((0, 3, 1, 2))
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(dim=0).cuda().half()

        rgb_video = np.array(input_tensor[0].cpu())
        rgb_video = normalize(rgb_video)
        rgb_video = rgb_video.transpose((0, 2, 3, 1))[:, :, :, ::-1]

        pred_class_idx = model(input_tensor).argmax().item()

        cam = cam_fn(
            model=model, target_layers=target_layers, use_cuda=not device == "cpu"
        )
        grayscale_cam = cam(input_tensor=input_tensor)

        vis_video = []
        for i in range(len(grayscale_cam)):
            visualization = show_cam_on_image(
                rgb_video[i], grayscale_cam[i], use_rgb=True
            )
            vis_video.append(visualization[:, :, ::-1])
        vis_video = np.array(vis_video)
        im_vis = ax_vis.imshow(normalize(vis_video[0, :, :, ::-1]))

        # Saving GIF with visualization
        fig.suptitle(
            "Truth : %s \nPrediction : %s" % (class_idx, pred_class_idx), fontsize=16
        )
        fig.tight_layout()
        fig.subplots_adjust(top=1)

        ax_vis.axis("off")
        ax_vis.set_title(f"{cam_fn.__name__} Visualization")

        ax_orig.axis("off")
        ax_orig.set_title(f"Original Video")

        def animate(i):
            im_orig.set_data(normalize(video_orig[i, :, :, ::-1]))
            im_vis.set_data(normalize(vis_video[i, :, :, ::-1]))
            return im_orig, im_vis

        anim = animation.FuncAnimation(
            fig, animate, frames=vis_video.shape[0], interval=10
        )
        save_path = os.path.join(outpath, f"vis_{idx}_{cam_fn.__name__}.gif")
        writergif = animation.PillowWriter(fps=30)
        anim.save(save_path, writer=writergif)

        plt.close()

    if verbose:
        print("Saved visualizations of indices %s at %s" % (str(indices), outpath))


if __name__ == "__main__":
    from utils.framesdata import FramesDataset
    from utils.translators import expts

    job_name = "GravityLvl2Combs_2022-04-04_20_48_54_771878"  # some job name for which you have a model

    # Function parameters
    dataset = FramesDataset(
        "data/", expts["gravity"], fpv=None, skip_every=1, train=True, shuffle=True
    )
    model = torch.load(f"output/{job_name}/model.pt")
    indices = np.random.choice(np.arange(0, len(dataset)), 12, replace=False)
    cam_fn = GradCAM  # Could be any of: GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
    outpath = f"output/{job_name}/gifs"

    # Running camify
    camify(dataset, model, indices, cam_fn, outpath, verbose=True)
