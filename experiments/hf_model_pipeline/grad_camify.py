import os
from glob import glob

from PIL import Image
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as f

from hf_utils import (
    find_best_ckpts,
    load_model_and_dataset,
    parse_args_evaluate,
    seed_everything,
)


def get_transform(preprocessor):
    mean = preprocessor.image_mean
    std = preprocessor.image_std
    if "shortest_edge" in preprocessor.size:
        height = width = preprocessor.size["shortest_edge"]
    else:
        height = preprocessor.size["height"]
        width = preprocessor.size["width"]
    resize_to = (height, width)

    transform = Compose(
        [
            ToTensor(),
            Normalize(mean, std),
            Resize(224, antialias=True),
        ]
    )
    return transform


def make_gif(images, filename):
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )


def timesformer_reshape_transform(tensor: torch.Tensor):
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    tensor = tensor[:, 1:, :].reshape(1, 8, 14, 14, 768)[:, 0]
    tensor = tensor.transpose(2, 3).transpose(1, 2)
    return tensor


def xclip_reshape_transform(tensor: torch.Tensor):
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    tensor = tensor[:, 1:-1, :].reshape(1, 8, 7, 7, 768)[:, 0]
    tensor = tensor.transpose(2, 3).transpose(1, 2)
    return tensor


if __name__ == "__main__":
    args = parse_args_evaluate()
    seed_everything()

    model_name, dataset_name = args.model_name, args.dataset_name
    ckpt_path = find_best_ckpts(model_name, f"{dataset_name}_data")
    model, video_dataset, _, is_rnn_mode = load_model_and_dataset(
        model_name=model_name,
        dataset_name=dataset_name,
        ckpt_path=ckpt_path,
        split=args.split,
    )

    preprocessor = model.preprocessor
    video_dataset.preprocess(model, is_rnn_mode)
    dataset = video_dataset.datasets[1][0]
    transform = get_transform(preprocessor)

    os.makedirs(f"cam_output/{model_name}/{dataset_name}", exist_ok=True)

    if model_name in {"resnet18", "densenet"}:
        model_base = model.model.embedder.model
        model_modules = [m for m in model_base.modules()]
        target_layers = [model_modules[-6]]

        video_name = next(dataset)["video_name"]
        video_frames_dir = video_name.replace(
            "experiment_video.mp4", "human_readable/frames/"
        )
        num_frames = len(os.listdir(video_frames_dir))

        cam = GradCAM(model=model_base, target_layers=target_layers, use_cuda=True)
        imgs = []
        for idx in tqdm(range(num_frames)):
            img_path = os.path.join(video_frames_dir, f"frame_{idx}.jpeg")
            if not os.path.isfile(img_path):
                img_path = img_path.replace("jpeg", "png")
            rgb_img = Image.open(img_path)
            rgb_img_normed = (
                np.float32(f.resize(rgb_img, (224, 224), antialias=True)) / 255.0
            )
            input_tensor = transform(rgb_img).unsqueeze(0)
            grayscale_cam = cam(input_tensor=input_tensor)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(
                rgb_img_normed, grayscale_cam, use_rgb=True
            )
            viz_img = Image.fromarray(visualization)
            viz_img.save(f"cam_output/{model_name}/{dataset_name}/{idx}.png")
            imgs.append(viz_img)

        out_gif_name = f"cam_output/{model_name}/{dataset_name}/full.gif"
        make_gif(imgs, out_gif_name)
        print("Saved at:", os.path.abspath(out_gif_name))

    if model_name == "timesformer":
        # model_base = model.model.timesformer.encoder
        model_base = model.model
        target_layers = [
            model.model.timesformer.encoder.layer[-1].layernorm_before,
            model.model.timesformer.encoder.layer[-2].layernorm_before,
        ]
        video_name = next(dataset)["video_name"]
        video_frames_dir = video_name.replace(
            "experiment_video.mp4", "human_readable/frames/"
        )
        num_frames = len(os.listdir(video_frames_dir))

        cam = GradCAM(
            model=model_base,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=timesformer_reshape_transform,
        )
        imgs = []
        cur_imgs = []
        for idx in tqdm(range(num_frames)):
            img_path = os.path.join(video_frames_dir, f"frame_{idx}.jpeg")
            if not os.path.isfile(img_path):
                img_path = img_path.replace("jpeg", "png")
            rgb_img = Image.open(img_path)
            rgb_img_normed = (
                np.float32(f.resize(rgb_img, (224, 224), antialias=True)) / 255.0
            )
            input_tensor = transform(rgb_img).unsqueeze(0).unsqueeze(0)
            cur_imgs.append(input_tensor)

            if idx == num_frames - 1:
                cur_imgs = cur_imgs + [cur_imgs[-1]] * (8 - len(cur_imgs))

            if len(cur_imgs) < 8:
                continue

            # input_tensor = input_tensor.expand((1, 8, 3, 224, 224))
            input_tensor = torch.cat(cur_imgs, axis=1)
            grayscale_cam = cam(input_tensor=input_tensor).transpose((1, 2, 0))
            visualization = show_cam_on_image(
                rgb_img_normed, grayscale_cam, use_rgb=True
            )
            viz_img = Image.fromarray(visualization)
            viz_img.save(f"cam_output/{model_name}/{dataset_name}/{idx}.png")
            imgs.append(viz_img)
            cur_imgs = []

        out_gif_name = f"cam_output/{model_name}/{dataset_name}/full.gif"
        make_gif(imgs, out_gif_name)
        print("Saved at:", os.path.abspath(out_gif_name))

    if model_name == "xclip":
        # model_base = model.model.timesformer.encoder
        model_base = model.model
        target_layers = [
            model.model.xclip.vision_model.encoder.layers[-2].layer_norm1,
            model.model.xclip.vision_model.encoder.layers[-1].layer_norm1,
        ]
        video_name = next(dataset)["video_name"]
        video_frames_dir = video_name.replace(
            "experiment_video.mp4", "human_readable/frames/"
        )
        num_frames = len(os.listdir(video_frames_dir))

        cam = GradCAM(
            model=model_base,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=xclip_reshape_transform,
        )
        imgs = []
        cur_imgs = []
        for idx in tqdm(range(num_frames)):
            img_path = os.path.join(video_frames_dir, f"frame_{idx}.jpeg")
            if not os.path.isfile(img_path):
                img_path = img_path.replace("jpeg", "png")
            rgb_img = Image.open(img_path)
            rgb_img_normed = (
                np.float32(f.resize(rgb_img, (224, 224), antialias=True)) / 255.0
            )
            input_tensor = transform(rgb_img).unsqueeze(0).unsqueeze(0)
            cur_imgs.append(input_tensor)

            if idx == num_frames - 1:
                cur_imgs = cur_imgs + [cur_imgs[-1]] * (8 - len(cur_imgs))

            if len(cur_imgs) < 8:
                continue

            # input_tensor = input_tensor.expand((1, 8, 3, 224, 224))
            input_tensor = torch.cat(cur_imgs, axis=1)
            grayscale_cam = cam(input_tensor=input_tensor).transpose((1, 2, 0))
            breakpoint()
            visualization = show_cam_on_image(
                rgb_img_normed, grayscale_cam, use_rgb=True
            )
            viz_img = Image.fromarray(visualization)
            viz_img.save(f"cam_output/{model_name}/{dataset_name}/{idx}.png")
            imgs.append(viz_img)
            cur_imgs = []

        out_gif_name = f"cam_output/{model_name}/{dataset_name}/full.gif"
        make_gif(imgs, out_gif_name)
        print("Saved at:", os.path.abspath(out_gif_name))
