# MMVGM 

This repository contains code for *fake video detection*, *fake video source tracing*, and *misuse prevention* tasks. We have proposed the first pipeline against the misuse and unsafe concern for video diffusion models.

This module contains:

1. Codes for trained detection and source tracing models with given data.
2. Approaches to use our misuse prevention strategy.

## Table of Contents

- [Download Dependencies](#download-dependencies)
	- [Video generation models dependencies](#video-generation-models-dependencies)
	- [Detection and Source tracing model dependencies](#detection-and-source-tracing-model-dependencies)
    - [I3D dependencies](#i3d-dependencies)
    - [X-Clip and VideoMAE dependencies](#x-clip-and-videomae-dependencies)
- [Model Training](#model-training)
- [Misuse Prevention](#misuse-prevention)

## Download Dependencies

### Video generation models dependencies

Our experiments include nine different generative tasks for each generation model environment. Please refer to their repository respectively: [Hotshot-xl](https://github.com/hotshotco/Hotshot-XL) [I2Vgen-xl](https://github.com/ali-vilab/i2vgen-xl) [Show-1](https://github.com/showlab/Show-1) [Videocrafter](https://github.com/AILab-CVC/VideoCrafter) [SEINE](https://github.com/Vchitect/SEINE) [LaVie](https://github.com/Vchitect/LaVie) [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) .


### Detection and Source tracing model dependencies

You can add the requirements files for the detection and source tracing models set up in this part.

#### I3D dependencies

Install [requirements_i3d.txt](detection_and_source_tracing/requirements_i3d.txt) and run:

```bash
pip install -r requirements_i3d.txt
```

#### X-Clip and VideoMAE dependencies

Install [requirements.txt](detection_and_source_tracing/requirements.txt) and run:

```bash
pip install -r requirements.txt
```

## Model Training

## Misuse Prevention
