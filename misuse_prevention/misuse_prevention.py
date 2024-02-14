import math
import os
from glob import glob
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
from lpips_pytorch import lpips,LPIPS
import clip
from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def get_loss(img_features, tar_features,img_v_features,tar_v_features, img, ori,lpips):
    alpha_1 = 1
    alpha_2 = 1
    alpha_3 = 0
    img_features = img_features.view(-1)
    tar_features = tar_features.view(-1)
    tar_v_features = tar_v_features.view(-1)
    img_v_features = img_v_features.view(-1)
    l2 = 1-F.cosine_similarity(img_features.unsqueeze(0),tar_features.unsqueeze(0),dim =1)
    l3 = 1-F.cosine_similarity(img_v_features.unsqueeze(0),tar_v_features.unsqueeze(0),dim =1)
    l1 = torch.abs(lpips(img,ori))
    return l1*alpha_1 + l2*alpha_2 + l3*alpha_3, l1,l2

def get_loss_untarget(img_features, tar_features,img_v_features,tar_v_features, img, ori,lpips):
    alpha_1 = 1
    alpha_2 = 1
    alpha_3 = 1
    img_features = img_features.view(-1)
    tar_features = tar_features.view(-1)
    tar_v_features = tar_v_features.view(-1)
    img_v_features = img_v_features.view(-1)
    l2 = 1-F.cosine_similarity(img_features.unsqueeze(0),tar_features.unsqueeze(0),dim =1)
    l3 = 1-F.cosine_similarity(img_v_features.unsqueeze(0),tar_v_features.unsqueeze(0),dim =1)
    l1 = torch.abs(lpips(img,ori))
    return l1*alpha_1 + l2*alpha_2 + l3*alpha_3, l1,l2



def misuse_prevention(
    input_path: str = "./test.png",  # Can either be image file or folder with image files
    tar_img_path: str = "./target.png",
    eps:float = 4/255,
    steps:int = 1000,
    directed: bool = True,
    save_dir: str = "./output.png",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda:2",
    output_folder: Optional[str] = None,
    ):
    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    torch.manual_seed(seed)
    
    # return
    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(device)
    tar_img = Image.open(tar_img_path).convert("RGB")
    tar_img = ToTensor()(tar_img)
    tar_img = tar_img * 2.0 - 1.0
    tar_img = tar_img.unsqueeze(0).to(device)
    tar_img_v = tar_img + 0.02 * torch.randn_like(tar_img)
    image_v = image + 0.02 * torch.randn_like(image)
    # print(model.conditioner.embedders)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)
    torch.cuda.set_device(2) 

    lpips = LPIPS(net_type ='vgg').to(device)
    image = image.clone().detach().requires_grad_(True)
    

    org_img = image.clone().detach().to(device)
    org_features = model.conditioner.embedders[0](org_img)
    org_img_v = org_img + 0.02 * torch.randn_like(org_img)
    org_v_features = model.conditioner.embedders[3](org_img_v)

    tar_features = model.conditioner.embedders[0](tar_img)
    tar_v_features = model.conditioner.embedders[3](tar_img_v)
    alpha = 1/255
    if directed:
        for step in range(steps):
            image.requires_grad_(True)
            img_features = model.conditioner.embedders[0](image)
            image_v = image + 0.02* torch.randn_like(image)
            img_v_features = model.conditioner.embedders[3](image_v)
            loss,l1,l2 = get_loss(img_features, tar_features, img_v_features,tar_v_features, image, org_img, lpips)
            grad = torch.autograd.grad(loss, image,
                                        retain_graph=False, create_graph=False)[0]
            image = image - alpha * grad.sign() 
            eta = torch.clamp(image - org_img, min=-eps, max=eps)
            image = torch.clamp(org_img + eta, min=-1, max=1).detach_()
            if step % 50 == 0:
                print(f"steps\t{step}\tloss:{loss}\tl1:{l1}\tl2:{l2}\t")
    else:
        for step in range(steps):
            image.requires_grad_(True)
            img_features = model.conditioner.embedders[0](image)
            image_v = image + 0.02* torch.randn_like(image)
            img_v_features = model.conditioner.embedders[3](image_v)
            loss,l1,l2 = get_loss_untarget(img_features, org_features, img_v_features,org_v_features, image, org_img, lpips)
            grad = torch.autograd.grad(loss, image,
                                        retain_graph=False, create_graph=False)[0]
            image = image + alpha * grad.sign() 
            eta = torch.clamp(image - org_img, min=-eps, max=eps)
            image = torch.clamp(org_img + eta, min=-1, max=1).detach_()
            if step % 50 == 0:
                print(f"steps\t{step}\tloss:{loss}\tl1:{l1}\tl2:{l2}\t")
    image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)

    to_pil = ToPILImage()
    pil_img = to_pil(image.squeeze(0))  

    pil_img.save(save_dir)



def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda:2":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda:2":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    Fire(misuse_prevention)
