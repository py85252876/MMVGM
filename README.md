# <img src="./utils/symbol.png" alt="symbol" style="height: 40px;"/> VGMShield: Mitigating Misuse of Video Generative Models

This repository contains code for fake video detection, fake video source tracing,
and misuse prevention tasks. It accompanies the VGMShield paper and keeps the
training/evaluation code lightweight by avoiding committed checkpoints and
legacy conda environment files.

<a href='https://arxiv.org/abs/2402.13126'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/pypy/VGMShield'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

## Setup With uv

This repo now uses `uv` instead of checked-in conda YAMLs. No lockfile is
committed, so the recommended workflow is `uv pip install -e ...`, which keeps
your platform-specific PyTorch wheel intact while resolving the latest
compatible project dependencies.

1. Create a virtual environment:

```bash
uv venv --python 3.11
```

2. Install the latest PyTorch build that matches your platform by following the
   official selector at https://pytorch.org/get-started/locally/.

Typical examples:

```bash
# macOS
uv pip install torch torchvision

# Linux + CUDA 12.8
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

3. Install the repo dependencies:

```bash
# Detection + source tracing
uv pip install -e ".[training]"

# Everything, including misuse-prevention helpers
uv pip install -e ".[training,misuse-prevention]"
```

4. Download the I3D RGB ImageNet checkpoint only if you need the I3D pipeline:

```bash
uv run python scripts/download_i3d_weights.py
```

## Video Generation Dependencies

The paper evaluates multiple video generation backbones. Refer to their upstream
repositories when you need the full generation environments:

- [Hotshot-XL](https://github.com/hotshotco/Hotshot-XL)
- [I2VGen-XL](https://github.com/ali-vilab/i2vgen-xl)
- [Show-1](https://github.com/showlab/Show-1)
- [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)
- [SEINE](https://github.com/Vchitect/SEINE)
- [LaVie](https://github.com/Vchitect/LaVie)
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models)

## Model Training

Run the commands below from the repository root.

> Note: The paper uses nine generation tasks for source tracing. Adjust the
> input paths and `--label_number` for your own setup.

- Training I3D-based detection:

```bash
uv run python detection_and_source_tracing/i3d.py \
  --train True \
  --task detection \
  --epoch 20 \
  --learning_rate 1e-5 \
  --pre_trained_I3D_model models/rgb_imagenet.pt \
  --fake_videos_path "path/to/fake/videos" \
  --real_videos_path "path/to/real/videos" \
  --label_number 2 \
  --save_checkpoint_dir checkpoints/i3d_detection.pt
```

- Training I3D-based source tracing:

```bash
uv run python detection_and_source_tracing/i3d.py \
  --train True \
  --task source_tracing \
  --epoch 20 \
  --learning_rate 1e-5 \
  --pre_trained_I3D_model models/rgb_imagenet.pt \
  --fake_videos_path "path/to/model_1" "path/to/model_2" \
  --label_number 2 \
  --save_checkpoint_dir checkpoints/i3d_source_tracing.pt
```

- Training VideoMAE-based detection:

```bash
uv run python detection_and_source_tracing/mae.py \
  --train True \
  --task detection \
  --epoch 20 \
  --learning_rate 1e-5 \
  --fake_videos_path "path/to/fake/videos" \
  --real_videos_path "path/to/real/videos" \
  --label_number 2 \
  --save_checkpoint_dir checkpoints/mae_detection.pt
```

- Training XCLIP-based detection:

```bash
uv run python detection_and_source_tracing/xclip.py \
  --train True \
  --task detection \
  --epoch 20 \
  --learning_rate 1e-5 \
  --fake_videos_path "path/to/fake/videos" \
  --real_videos_path "path/to/real/videos" \
  --label_number 2 \
  --save_checkpoint_dir checkpoints/xclip_detection.pt
```

All three training/evaluation scripts also accept:

- `--device` to choose the target device, for example `cuda:0` or `cpu`
- `--batch_size` to override the default batch size of `4`
- `--hf_cache_dir` if you want Hugging Face assets in a non-default cache path

## Model Evaluation

We provide trained detection and source-tracing checkpoints in the
[Hugging Face repository](https://huggingface.co/pypy/VGMShield).

- Evaluate I3D:

```bash
uv run python detection_and_source_tracing/i3d.py \
  --train False \
  --task detection \
  --load_pre_trained_model_state path/to/checkpoint.pt \
  --fake_videos_path "path/to/fake/videos" \
  --real_videos_path "path/to/real/videos" \
  --label_number 2
```

- Evaluate VideoMAE:

```bash
uv run python detection_and_source_tracing/mae.py \
  --train False \
  --task detection \
  --load_pre_trained_model_state path/to/checkpoint.pt \
  --fake_videos_path "path/to/fake/videos" \
  --real_videos_path "path/to/real/videos" \
  --label_number 2
```

- Evaluate XCLIP:

```bash
uv run python detection_and_source_tracing/xclip.py \
  --train False \
  --task detection \
  --load_pre_trained_model_state path/to/checkpoint.pt \
  --fake_videos_path "path/to/fake/videos" \
  --real_videos_path "path/to/real/videos" \
  --label_number 2
```

## Misuse Prevention

The misuse-prevention pipeline depends on modules from
[Stable Video Diffusion](https://github.com/Stability-AI/generative-models).
This repository does not vendor those modules, so make sure that checkout is
available on `PYTHONPATH`, and either pass `--svd_root` or set
`STABLE_VIDEO_DIFFUSION_ROOT`, before running the script.

Directed defense:

```bash
PYTHONPATH=/path/to/generative-models:$PYTHONPATH \
uv run python misuse_prevention/misuse_prevention.py \
  --svd_root /path/to/generative-models \
  --input_path original_image \
  --tar_img_path target_image \
  --steps iteration_steps \
  --eps 4/255
```

Undirected defense:

```bash
PYTHONPATH=/path/to/generative-models:$PYTHONPATH \
uv run python misuse_prevention/misuse_prevention.py \
  --svd_root /path/to/generative-models \
  --input_path original_image \
  --directed False \
  --steps iteration_steps \
  --eps 4/255
```

## 🖊️ Citation

```BibTex
@misc{pang2024vgmshield,
      title={VGMShield: Mitigating Misuse of Video Generative Models}, 
      author={Yan Pang and Yang Zhang and Tianhao Wang},
      year={2024},
      eprint={2402.13126},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

## 🥰 Acknowledgement

We feel gratitude for the previous open-source work that helped us construct our **VGMShield**. These works include but are not limited to [Video Features](https://github.com/v-iashin/video_features), [VideoX](https://github.com/microsoft/VideoX),[Hotshot-xl](https://github.com/hotshotco/Hotshot-XL), [I2Vgen-xl](https://github.com/ali-vilab/i2vgen-xl), [Show-1](https://github.com/showlab/Show-1), [Videocrafter](https://github.com/AILab-CVC/VideoCrafter), [SEINE](https://github.com/Vchitect/SEINE), [LaVie](https://github.com/Vchitect/LaVie), and [Stable Video Diffusion](https://github.com/Stability-AI/generative-models). We respect their effort and original contributions.
