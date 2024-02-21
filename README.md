# <img src="./utils/symbol.png" alt="symbol" style="height: 32px;"/> VGMShield: Mitigating Misuse of Video Generative Models

This repository contains code for *fake video detection*, *fake video source tracing*, and *misuse prevention* tasks. We have proposed the first pipeline against the misuse and unsafe concern for video diffusion models.

This module contains:

1. Codes for trained detection and source tracing models with given data.
2. Approaches to use our misuse prevention strategy.

## 📄 Table of Contents

- [📄 Table of Contents](#table_of_contents)
- [🛠️ Download Dependencies](#download-dependencies)
	- [Video generation models dependencies](#video-generation-models-dependencies)
	- [Detection and Source tracing model dependencies](#detection-and-source-tracing-model-dependencies)
    - [I3D dependencies](#i3d-dependencies)
    - [X-Clip and VideoMAE dependencies](#x-clip-and-videomae-dependencies)
- [🚀 Model Training](#model-training)
- [💪 Misuse Prevention](#misuse-prevention)
- [🖊️ Citation](#citation)

## 🛠️ Download Dependencies

### Video generation models dependencies

Our experiments include nine different generative tasks for each generation model environment. Please refer to their repository respectively: [Hotshot-xl](https://github.com/hotshotco/Hotshot-XL) [I2Vgen-xl](https://github.com/ali-vilab/i2vgen-xl) [Show-1](https://github.com/showlab/Show-1) [Videocrafter](https://github.com/AILab-CVC/VideoCrafter) [SEINE](https://github.com/Vchitect/SEINE) [LaVie](https://github.com/Vchitect/LaVie) [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) .


### Detection and Source tracing model dependencies

You can add the requirements files for the detection and source tracing models in this part.

#### I3D dependencies

Install [environment_i3d.yml](utils/requirement/environment_i3d.yml) and run:

```bash
conda env create -f environment_i3d.yml

```

#### X-Clip and VideoMAE dependencies

Install [environment_mae.yml](utils/requirement/environment_mae.yml) and run:

```bash
conda env create -f environment_mae.yml
```

## 🚀 Model Training

Train detection and source tracing model based on the I3D model.

First, enter [detection and source tracing directory](direction_and_source_tracing)

```bash
cd direction_and_source_tracing
```

> Note: The default setting for source tracing is the nine generation tasks we mentioned in our paper. Please change the code for your own tasks.

- **Training I3D-based detection model**

```bash
python i3d.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./test.pt \
    --task "detection" \
    --pre_trained_I3D_model ../models/rgb_imagenet.pt --fake_videos_path \
    --real_videos_path ./invid/clip --label_number 2
```

- **Training I3D-based source tracing model**

```bash
python i3d.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./test.pt \
    --task "source_tracing" \
    --pre_trained_I3D_model ../models/rgb_imagenet.pt --fake_videos_path \
    "./Hotshot-XL/outputs/invid" \
    "./i2vgen-xl/outputs/invid/i2v" \
    "./i2vgen-xl/outputs/invid/t2v" \
    "./LaVie/res/base/invid" \
    "./SEINE/results/invid/i2v" \
    "./Show-1/outputs/invid" \
    "./video_prevention/outputs/Invid/svd_xt" \
    "./VideoCrafter/results/invid/i2v" \
    "./VideoCrafter/results/invid/t2v"
```


Develop the detection and source tracing model using VideoMAE as the backbone.

- **Training MAE-based detection model**

```bash
python mae.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./test.pt \
    --task "detection" \
    --fake_videos_path "./Hotshot-XL/outputs/invid" \
    --real_videos_path ./invid/clip --label_number 2
```

- **Training MAE-based source tracing model**

```bash
python mae.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./test.pt \
    --task "source_tracing" \
    --fake_videos_path "./Hotshot-XL/outputs/invid" \
    "./i2vgen-xl/outputs/invid/i2v" \
    "./i2vgen-xl/outputs/invid/t2v" \
    "./LaVie/res/base/invid" \
    "./SEINE/results/invid/i2v" \
    "./Show-1/outputs/invid" \
    "./video_prevention/outputs/Invid/svd_xt" \
    "./VideoCrafter/results/invid/i2v" \
    "./VideoCrafter/results/invid/t2v"
```

Build the detection and source tracing model using xclip.

- **Training XCLIP-based detection model**

```bash
python xclip.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./test.pt \
    --task "detection" \
    --fake_videos_path "./Hotshot-XL/outputs/invid" \
    --real_videos_path ./invid/clip --label_number 2
```

- **Training XCLIP-based source tracing model**

```bash
python xclip.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./test.pt \
    --task "source_tracing" \
    --fake_videos_path "./Hotshot-XL/outputs/invid" \
    "./i2vgen-xl/outputs/invid/i2v" \
    "./i2vgen-xl/outputs/invid/t2v" \
    "./LaVie/res/base/invid" \
    "./SEINE/results/invid/i2v" \
    "./Show-1/outputs/invid" \
    "./video_prevention/outputs/Invid/svd_xt" \
    "./VideoCrafter/results/invid/i2v" \
    "./VideoCrafter/results/invid/t2v"
```

## 💪 Misuse Prevention

We provided two defense strategies, which are *directed defense* and *undirected defense*. To execute the *directed defense* approach, run:

```bash
python misuse_prevention.py --input_path original_image --tar_img_path target_image --steps iteration_steps --eps 4/255
```

For *undirected defense*, run:

```bash
python misuse_prevention.py --input_path original_image --directed False --steps iteration_steps --eps 4/255
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

