# PhenoIMC3D: Iterative Motion Compensation for Canonical 3D Reconstruction (RA-L 2026)

[Andre Rochow\*](https://github.com/andrerochow), [Jonas Marcic\*](https://github.com/jonasmrc), [Svetlana Seliunina](https://github.com/SvetlanaSeliunina), and [Sven Behnke](https://www.ais.uni-bonn.de/behnke/)  
\* equal contribution

#### [[Paper](https://ieeexplore.ieee.org/abstract/document/11447388)] | [[arXiv](https://arxiv.org/abs/2510.15491)] | [[Dataset](https://dataverse.harvard.edu/previewurl.xhtml?token=732e2bf6-8d61-4463-b48c-bb86a074adce)]

---

| Without Motion Compensation | PhenoIMC3D (ours) |
|---------------------------|-----------|
| ![](assets/DJI_0104-0.png) | ![](assets/DJI_0104-100.png) |

---

## Description
Official implementation of *Iterative Motion Compensation for Canonical 3D Reconstruction From UAV Plant Images Captured in Windy Conditions* (accepted to IEEE Robotics and Automation Letters, 2026) by Andre Rochow*, Jonas Marcic*, Svetlana Seliunina, and Sven Behnke.

The method models and compensates unwanted motion in scenes captured by UAVs, such as plant movement caused by downwash. These effects can significantly degrade the quality of 3D reconstruction.

PhenoIMC3D enables:
- motion compensation in dynamic scenes  
- extraction of a single sharp canonical 3D representation 
- compatibility with various 3D reconstruction methods  

PhenoIMC3D can be seamlessly combined with:
- NeRF-based methods  
- 3D Gaussian Splatting  
- other recent splatting-based methods  

To integrate PhenoIMC3D with your reconstruction pipeline, follow the instructions in the *Setup* and *Configuration* sections.

---

## Setup
Choose and install a 3D reconstruction method of your choice.

We provide examples for:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Deformable Beta Splatting](https://github.com/RongLiu-Leo/beta-splatting)  

However, any other reconstruction method can also be used.

### Clone the repository

```bash
git clone https://github.com/AIS-Bonn/PhenoIMC3D.git --recursive
cd PhenoIMC3D/submodules/RAFT
./download_models.sh
```

### Install dependencies

Activate the environment of your selected reconstruction method:

```bash
conda activate your_3d_method
```

Install required packages **(if not already available in your environment)**:

```bash
pip install torch torchvision
pip install scipy
pip install opencv-python
pip install tqdm
pip install natsort
```

---

## Configuration

We provide configuration files for integrating PhenoIMC3D with:

- 3D Gaussian Splatting  
- Deformable Beta Splatting  

You can also define configurations for custom reconstruction methods.

### Placeholders

| Placeholder        | Description |
|------------------|------------|
| `${dataset_path}` | Path to the dataset (provided via CLI arguments) |
| `${model_path}`   | Path to the output model/workspace directory |

These placeholders are automatically replaced at runtime.

---

### 3D Gaussian Splatting  
`configs/gaussian_splatting.json`

```json
{
    "train_cmd": "python /path/to/your/installed/gaussian-splatting/train.py -s ${dataset_path} -m ${model_path} -r 1",
    "model_weights": "${model_path}/point_cloud/iteration_30000/point_cloud.ply",
    "render_cmd": "python /path/to/your/installed/gaussian-splatting/render.py -m ${model_path} --skip_test",
    "img_predictions": "${model_path}/train/ours_30000/renders"
}
```

---

### Deformable Beta Splatting  
`configs/beta_splatting.json`

```json
{
    "train_cmd": "python /path/to/your/installed/beta-splatting/train.py -s ${dataset_path} --model_path ${model_path} --resolution 1 --iterations 30000 --disable_viewer",
    "model_weights": "${model_path}/point_cloud/iteration_30000/point_cloud.ply",
    "render_cmd": "python /path/to/your/installed/beta-splatting/render.py -m ${model_path} --skip_test",
    "img_predictions": "${model_path}/train/ours_30000/renders"
}
```

---

### Custom 3D Reconstruction Method

```json
{
    "train_cmd": "COMMAND TO RUN TRAINING OF YOUR RECONSTRUCTION METHOD",
    "model_weights": "OPTIONAL: PATH TO MODEL CHECKPOINTS",
    "render_cmd": "COMMAND TO RUN RENDERING",
    "img_predictions": "PATH TO PREDICTED IMAGES"
}
```

---

## Prepare Dataset
To use our dataset, download it via the link above. It contains high-resolution images (8064×6048), so you may want to downscale them before training. A script is provided for this purpose.
Note that, due to [RAFT](https://github.com/princeton-vl/RAFT), both height and width must be divisible by 8.
```bash
python scale_dataset.py --dataset_path /path/to/datset/scene --colmap_binary /path/to/your/colmap/binary --r resolution
```
In our paper, we used `--r 4`, corresponding to a resolution of (2016×1512).

---

## Training

### 3D Gaussian Splatting

```bash
conda activate gaussian_splatting

python train.py \
    --config configs/gaussian_splatting.json \
    --dataset_path /path/to/dataset/scene \
    --model_path output/workspace/model_path
```

---

### Deformable Beta Splatting

```bash
conda activate beta_splatting

python train.py \
    --config configs/beta_splatting.json \
    --dataset_path /path/to/dataset/scene \
    --model_path output/workspace/model_path
```

---

## Some Results

| Without Motion Compensation | PhenoIMC3D (ours) |
|---------------------------|-----------|
| ![](assets/DJI_0540-0.png) | ![](assets/DJI_0540-100.png) |
| ![](assets/DJI_0418-0.png) | ![](assets/DJI_0418-100.png) |

For additional results and a quantitative evaluation, please refer to our paper.

---

## Citation

If you use this work, please cite:

```bibtex
@article{rochow2026iterative,
  title={Iterative Motion Compensation for Canonical 3D Reconstruction From UAV Plant Images Captured in Windy Conditions},
  author={Rochow, Andre and Marcic, Jonas and Seliunina, Svetlana and Behnke, Sven},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  publisher={IEEE}
}
```
