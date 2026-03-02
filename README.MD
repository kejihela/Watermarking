##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)

# Implementation of "FeatMark: Feature-level Watermark Protection against Mimicry Attacks with Diffusion Models"

## Environment setup

Install dependencies:
```shell
cd watermarks
conda create -n featmark python=3.9  
conda activate featmark
pip install -r requirements.txt  
pip install "huggingface-hub==0.25.2"
```

> GPU allocation: All experiments are performed on a single NVIDIA 80GB A100X GPU.


## Dataset preparation

This part is adopted from [Anti-DreamBooth](https://anti-dreambooth.github.io/).


We have experimented on these two datasets:
- VGGFace2: contains around 3.31 million images of 9131 person identities. We only use subjects that have at least 15 images of resolution above $500 \times 500$.
- CelebA-HQ: consists of 30,000 images at $1024 × 1024$ resolution. We
use the annotated subset from [here](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) that filters and groups images into 307 subjects with at least 15 images for each subject.

In this research, we select 50 identities from each dataset and carefully choose a subset of 12 images for each individual based on good pose and lighting. These examples are evenly divided into 3 subsets, including the reference clean set (set A), the target projecting set (set B), and an extra clean set for uncontrolled setting experiments (set C). These full split sets of each dataset are provided at [here](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV).

For convenient testing, we have provided a split set of one subject in VGGFace2 at `./data/n000050/`.

After downloading the dataset, you need to put the dataset in the `./data` folder. The code will access personal data according to directories like `./data/CelebA-HQ/5/set_A` and `./data/VGGFace2/n000050/set_A`.

## How to run
Please see the bash scripts in `./scripts` folder.