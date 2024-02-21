# <img src="./utils/symbol.png" alt="symbol" style="height: 40px;"/> VGMShield: Mitigating Misuse of Video Generative Models


This repository contains code for *fake video detection*, *fake video source tracing*, and *misuse prevention* tasks. We have proposed the first pipeline against the misuse and unsafe concern for video diffusion models.

<a href='https://arxiv.org/abs/2402.13126'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/pypy/VGMShield'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> 
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

This repository contains:

1. Codes for trained detection and source tracing models with given data.
2. Introduce how to evaluate detection source tracing models on given/custom data.
3. Approaches to use our misuse prevention strategy.

## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🛠️ Download Dependencies](#-download-dependencies)
	- [Video generation models dependencies](#video-generation-models-dependencies)
	- [Detection and Source tracing model dependencies](#detection-and-source-tracing-model-dependencies)
	    - [I3D dependencies](#i3d-dependencies)
	    - [X-CLIP and VideoMAE dependencies](#x-clip-and-videomae-dependencies)
- [🚀 Model Training](#-model-training)
- [👀 Model Evaluation](#-model-evaluation)
- [💪 Misuse Prevention](#-misuse-prevention)
- [🖊️ Citation](#-citation)
- [🥰 Acknowledgement](#-acknowledgement)

## 🛠️ Download Dependencies

### Video generation models dependencies

Our experiments include nine different generative tasks for each generation model environment. Please refer to their repository respectively: [Hotshot-xl](https://github.com/hotshotco/Hotshot-XL) [I2Vgen-xl](https://github.com/ali-vilab/i2vgen-xl) [Show-1](https://github.com/showlab/Show-1) [Videocrafter](https://github.com/AILab-CVC/VideoCrafter) [SEINE](https://github.com/Vchitect/SEINE) [LaVie](https://github.com/Vchitect/LaVie) [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) .


### Detection and Source tracing model dependencies

In this part, you can set up the environment for detection and source tracing models.

#### I3D dependencies

Install [environment_i3d.yml](utils/requirement/environment_i3d.yml) and run:

```bash
conda env create -f environment_i3d.yml

```

#### X-CLIP and VideoMAE dependencies

Install [environment_mae.yml](utils/requirement/environment_mae.yml) and run:

```bash
conda env create -f environment_mae.yml
```

## 🚀 Model Training

This part provides instructions on how to train different backbone detection and source tracing models.

First, enter [detection and source tracing directory](./detection_and_source_tracing)

```bash
cd direction_and_source_tracing
```

> Note: The default setting for source tracing is the nine generation tasks as we mentioned in our paper. Please change the code for your own tasks.

- **Training I3D-based detection model**

```bash
python i3d.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./save.pt \
    --task "detection" \
    --pre_trained_I3D_model ../models/rgb_imagenet.pt --fake_videos_path "fake videos' path" \
    --real_videos_path "real videos' path" --label_number 2
```

- **Training I3D-based source tracing model**

```bash
python i3d.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./save.pt \
    --task "source_tracing" \
    --pre_trained_I3D_model ../models/rgb_imagenet.pt --fake_videos_path \
    "fake videos generated from model 1" \
    ...
    "fake videos generated from model 9" --label_number 9
```


Develop the detection and source tracing model using VideoMAE as the backbone.

- **Training MAE-based detection model**

```bash
python mae.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./save.pt \
    --task "detection" \
    --fake_videos_path "fake videos' path" \
    --real_videos_path "real videos' path" --label_number 2
```

- **Training MAE-based source tracing model**

```bash
python mae.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./save.pt \
    --task "source_tracing" \
    --fake_videos_path \
    "fake videos generated from model 1" \
    ...
    "fake videos generated from model 9" --label_number 9
```

Build the detection and source tracing model using XCLIP.

- **Training XCLIP-based detection model**

```bash
python xclip.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./save.pt \
    --task "detection" \
    --fake_videos_path "fake videos' path" \
    --real_videos_path "real videos' path" --label_number 2
```

- **Training XCLIP-based source tracing model**

```bash
python xclip.py --train True --epoch 20 --learning_rate 1e-5 --save_checkpoint_dir ./save.pt \
    --task "source_tracing" \
    --fake_videos_path \
    "fake videos generated from model 1" \
    ...
    "fake videos generated from model 9" --label_number 9
```

## 👀 Model Evaluation

After training the detection and source tracing model, we can test our model's performance here.

> We have provided pre-trained detection and source tracing checkpoints at our [🤗 huggingface repository](https://huggingface.co/pypy/VGMShield). Please feel free to use it.

- **Testing I3D-based detection model**

```bash
python i3d.py --train False --task "detection" \
    --load_pre_trained_model_state "Your pre-trained model's path" --fake_videos_path \
    "fake video path" \
    --real_videos_path "real video path" --label_number 2
```

- **Testing I3D-based source tracing model**

```bash
python i3d.py --train False --task "source_tracing" \
    --load_pre_trained_model_state "Your pre-trained model's path" --fake_videos_path \
    "fake videos generated from model 1" \
    ...
    "fake videos generated from model 9" --label_number 9
```

- **Testing MAE-based detection model**

```bash
python mae.py --train False --task "detection" \
    --load_pre_trained_model_state "Your pre-trained model's path" --fake_videos_path \
    "fake video path" \
    --real_videos_path "real video path" --label_number 2
```
  
- **Testing MAE-based source tracing model**

```bash
python mae.py --train False --task "source_tracing" \
    --load_pre_trained_model_state "Your pre-trained model's path" --fake_videos_path \
    "fake videos generated from model 1" \
    ...
    "fake videos generated from model 9" --label_number 9
```

- **Testing XCLIP-based detection model**

```bash
python xclip.py --train False --task "detection" \
    --load_pre_trained_model_state "Your pre-trained model's path" --fake_videos_path \
    "fake video path" \
    --real_videos_path "real video path" --label_number 2
```
  
- **Testing XCLIP-based source tracing model**

```bash
python xclip.py --train False --task "source_tracing" \
    --load_pre_trained_model_state "Your pre-trained model's path" --fake_videos_path \
    "fake videos generated from model 1" \
    ...
    "fake videos generated from model 9" --label_number 9
```

## 💪 Misuse Prevention

> Note: In our paper, we use the encoders from [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) to add perturbation. The environment for our defense methods is the same as [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

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

## 🥰 Acknowledgement

We feel gratitude for the previous open-source work that helped us construct our **VGMShield**. These works include but are not limited to [Video Features](https://github.com/v-iashin/video_features), [VideoX](https://github.com/microsoft/VideoX),[Hotshot-xl](https://github.com/hotshotco/Hotshot-XL), [I2Vgen-xl](https://github.com/ali-vilab/i2vgen-xl), [Show-1](https://github.com/showlab/Show-1), [Videocrafter](https://github.com/AILab-CVC/VideoCrafter), [SEINE](https://github.com/Vchitect/SEINE), [LaVie](https://github.com/Vchitect/LaVie), and [Stable Video Diffusion](https://github.com/Stability-AI/generative-models). We respect their effort and original contributions.

