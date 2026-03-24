from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _runtime() -> SimpleNamespace:
    try:
        from einops import repeat
        from fire import Fire
        from lpips_pytorch import LPIPS
        from omegaconf import OmegaConf
        from PIL import Image
        from torchvision.transforms import ToPILImage, ToTensor

        from scripts.util.detection.nsfw_and_watermark_dectection import (
            DeepFloydDataFiltering,
        )
        from sgm.util import default, instantiate_from_config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "misuse_prevention requires `uv pip install -e '.[misuse-prevention]'` and "
            "the Stable Video Diffusion checkout on PYTHONPATH."
        ) from exc

    return SimpleNamespace(
        DeepFloydDataFiltering=DeepFloydDataFiltering,
        Fire=Fire,
        Image=Image,
        LPIPS=LPIPS,
        OmegaConf=OmegaConf,
        ToPILImage=ToPILImage,
        ToTensor=ToTensor,
        default=default,
        instantiate_from_config=instantiate_from_config,
        repeat=repeat,
    )


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({embedder.input_key for embedder in conditioner.embedders})


def get_batch(keys, value_dict, batch_shape, num_video_frames, device):
    runtime = _runtime()
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(batch_shape)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(batch_shape)))
            )
        elif key == "cond_aug":
            batch[key] = runtime.repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(batch_shape),
            )
        elif key == "cond_frames":
            batch[key] = runtime.repeat(value_dict["cond_frames"], "1 ... -> b ...", b=batch_shape[0])
        elif key == "cond_frames_without_noise":
            batch[key] = runtime.repeat(
                value_dict["cond_frames_without_noise"],
                "1 ... -> b ...",
                b=batch_shape[0],
            )
        else:
            batch[key] = value_dict[key]

    if num_video_frames is not None:
        batch["num_video_frames"] = num_video_frames

    for key, value in batch.items():
        if key not in batch_uc and isinstance(value, torch.Tensor):
            batch_uc[key] = value.clone()
    return batch, batch_uc


def get_loss(img_features, tar_features, img_v_features, tar_v_features, img, ori, lpips_model):
    img_features = img_features.view(-1)
    tar_features = tar_features.view(-1)
    tar_v_features = tar_v_features.view(-1)
    img_v_features = img_v_features.view(-1)
    l2 = 1 - F.cosine_similarity(img_features.unsqueeze(0), tar_features.unsqueeze(0), dim=1)
    l3 = 1 - F.cosine_similarity(img_v_features.unsqueeze(0), tar_v_features.unsqueeze(0), dim=1)
    l1 = torch.abs(lpips_model(img, ori))
    return l1 + l2 + l3, l1, l2


def get_loss_untarget(img_features, tar_features, img_v_features, tar_v_features, img, ori, lpips_model):
    img_features = img_features.view(-1)
    tar_features = tar_features.view(-1)
    tar_v_features = tar_v_features.view(-1)
    img_v_features = img_v_features.view(-1)
    l2 = 1 - F.cosine_similarity(img_features.unsqueeze(0), tar_features.unsqueeze(0), dim=1)
    l3 = 1 - F.cosine_similarity(img_v_features.unsqueeze(0), tar_v_features.unsqueeze(0), dim=1)
    l1 = torch.abs(lpips_model(img, ori))
    return l1 + l2 + l3, l1, l2


def misuse_prevention(
    input_path: str = "./test.png",
    tar_img_path: str = "./target.png",
    eps: float = 4 / 255,
    steps: int = 1000,
    directed: bool = True,
    save_dir: str = "./output.png",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    seed: int = 23,
    device: str = "cuda:0",
    output_folder: Optional[str] = None,
    svd_root: Optional[str] = None,
):
    runtime = _runtime()
    num_frames = runtime.default(num_frames, 14)
    num_steps = runtime.default(num_steps, 25)
    output_folder = runtime.default(output_folder, "outputs/simple_video_sample/svd/")
    _ = version, output_folder
    model_config = resolve_model_config("scripts/sampling/configs/svd.yaml", svd_root)

    model, _filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    torch.manual_seed(seed)

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and device_obj.index is not None:
        torch.cuda.set_device(device_obj.index)

    path = Path(input_path)
    with runtime.Image.open(path) as image:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        width, height = image.size

        if height % 64 != 0 or width % 64 != 0:
            resized_width, resized_height = map(lambda x: x - x % 64, (width, height))
            image = image.resize((resized_width, resized_height))
            print(
                f"WARNING: Your image is of size {height}x{width} which is not divisible by 64. "
                f"We are resizing to {resized_height}x{resized_width}!"
            )

        image = runtime.ToTensor()(image)
        image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(device)
    tar_img = runtime.Image.open(tar_img_path).convert("RGB")
    tar_img = runtime.ToTensor()(tar_img)
    tar_img = tar_img * 2.0 - 1.0
    tar_img = tar_img.unsqueeze(0).to(device)
    tar_img_v = tar_img + 0.02 * torch.randn_like(tar_img)

    lpips_model = runtime.LPIPS(net_type="vgg").to(device)
    image = image.clone().detach().requires_grad_(True)

    org_img = image.clone().detach().to(device)
    org_features = model.conditioner.embedders[0](org_img)
    org_img_v = org_img + 0.02 * torch.randn_like(org_img)
    org_v_features = model.conditioner.embedders[3](org_img_v)

    tar_features = model.conditioner.embedders[0](tar_img)
    tar_v_features = model.conditioner.embedders[3](tar_img_v)
    alpha = 1 / 255
    if directed:
        for step in range(steps):
            image.requires_grad_(True)
            img_features = model.conditioner.embedders[0](image)
            image_v = image + 0.02 * torch.randn_like(image)
            img_v_features = model.conditioner.embedders[3](image_v)
            loss, l1, l2 = get_loss(
                img_features,
                tar_features,
                img_v_features,
                tar_v_features,
                image,
                org_img,
                lpips_model,
            )
            grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
            image = image - alpha * grad.sign()
            eta = torch.clamp(image - org_img, min=-eps, max=eps)
            image = torch.clamp(org_img + eta, min=-1, max=1).detach_()
            if step % 50 == 0:
                print(f"steps\t{step}\tloss:{loss}\tl1:{l1}\tl2:{l2}\t")
    else:
        for step in range(steps):
            image.requires_grad_(True)
            img_features = model.conditioner.embedders[0](image)
            image_v = image + 0.02 * torch.randn_like(image)
            img_v_features = model.conditioner.embedders[3](image_v)
            loss, l1, l2 = get_loss_untarget(
                img_features,
                org_features,
                img_v_features,
                org_v_features,
                image,
                org_img,
                lpips_model,
            )
            grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
            image = image + alpha * grad.sign()
            eta = torch.clamp(image - org_img, min=-eps, max=eps)
            image = torch.clamp(org_img + eta, min=-1, max=1).detach_()
            if step % 50 == 0:
                print(f"steps\t{step}\tloss:{loss}\tl1:{l1}\tl2:{l2}\t")
    image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)

    to_pil = runtime.ToPILImage()
    pil_img = to_pil(image.squeeze(0))
    pil_img.save(save_dir)


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    runtime = _runtime()
    config = runtime.OmegaConf.load(config)
    device_obj = torch.device(device)

    if device_obj.type == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    model = runtime.instantiate_from_config(config.model).to(device).eval()
    filter_model = runtime.DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter_model


def resolve_model_config(config_path: str, svd_root: Optional[str]) -> str:
    path = Path(config_path)
    if path.exists():
        return str(path)

    root = svd_root or os.environ.get("STABLE_VIDEO_DIFFUSION_ROOT")
    if root:
        candidate = Path(root) / config_path
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find the Stable Video Diffusion config. Pass --svd_root "
        "or set STABLE_VIDEO_DIFFUSION_ROOT to the upstream checkout."
    )


def main():
    runtime = _runtime()
    runtime.Fire(misuse_prevention)


if __name__ == "__main__":
    main()
