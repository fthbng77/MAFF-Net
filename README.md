<div align="center">  

# MAFF-Net: Enhancing 3D Object Detection with 4D Radar via Multi-Assist Feature Fusion

</div>
<div align="center">   

[![IEEE RA-L](https://img.shields.io/badge/IEEE%20RA--L-PDF-blue?style=flat&logo=IEEE&logoColor=white)](https://ieeexplore.ieee.org/document/10923711/)

</div>

## 📰 News

- 📢 **2025.5.30** : The code is uploaded. Please stay tuned for updates. 

- 🔔 **2025.3.12**：[Early Access](https://ieeexplore.ieee.org/document/10923711/)

- ✨ **2025.2.27**: RA-L Accepted

## 📝 Abstract

Perception systems are crucial for the safe operation of autonomous vehicles, particularly for 3D object detection. While LiDAR-based methods are limited by adverse weather conditions, 4D radars offer promising all-weather capabilities. However, 4D radars introduce challenges such as extreme sparsity, noise, and limited geometric information in point clouds. To address these issues, we propose MAFF-Net, a novel multi-assist feature fusion network specifically designed for 3D object detection using a single 4D radar. We introduce a sparsity pillar attention (SPA) module to mitigate the effects of sparsity while ensuring a sufficient receptive field. Additionally, we design the cluster query cross-attention (CQCA) module, which uses velocity-based clustered features as queries in the cross-attention fusion process. This helps the network enrich feature representations of potential objects while reducing measurement errors caused by angular resolution and multipath effects. Furthermore, we develop a cylindrical denoising assist (CDA) module to reduce noise interference, improving the accuracy of 3D bounding box predictions. Experiments on the VoD and TJ4DRadSet datasets demonstrate that MAFF-Net achieves state-of-the-art performance, outperforming 16-layer LiDAR systems and operating at over 17.9 FPS, making it suitable for real-time detection in autonomous vehicles.

## ⚙️ Method

![Overall framework](./docs/pics/method.jpg)

***Overview of the proposed MAFF-Net.*** *MAFF-Net consists of three components: the main branch, the assisted branch, and the detection head.
In the main branch, we apply sparse pillar attention (SPA) to the BEV features generated from the raw point cloud using a pillar-based method, ensuring global interaction and a sufficient receptive field.
The assisted branch introduces clustering query cross-attention (CQCA), using clustering feature assistance (CFA) to generate BEV queries for cross-attention fusion (CAF), which helps reduce noise and identify potential objects. We also design cylindrical denoising assistance (CDA), a sampling strategy inspired by cylindrical constraints, to filter noise and background points using the proposal's positional information.
Finally, fused BEV features are aggregated with clustered point cloud features at the keypoints' locations, and a multi-task detection head predicts the 3D bounding boxes.*

## 📜 Getting Started

step 1. Refer to [install.md](./docs/guidance/Install.md) to install the environment.

step 2. Refer to [dataset.md](./docs/guidance/dataset.md) to prepare View-of-delft (VoD) and TJ4DRadSet datasets.

step 3. Refer to [train_and_test.md](./docs/guidance/train_and_test.md) for training and testing.

## 📊 Model Zoo

We offer the model on VoD and TJ4DRadset.

|   Dataset    |                            Config                            |                                        Model Weights                                         | 
|:------------:|:------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
|     VoD      |  [MAFF-Net_vod.yaml](tools/cfgs/MAFF-Net/MAFF-Net_vod.yaml)  | [Link](https://github.com/TRV-Lab/MAFF-Net/releases/download/checkpoints/vod2025010810360.pth) |
|  TJ4DRadSet  | [MAFF-Net_TJ4D.yaml](tools/cfgs/MAFF-Net/MAFF-Net_TJ4D.yaml) | [Link](https://github.com/TRV-Lab/MAFF-Net/releases/download/checkpoints/tj4d2025060814210.pth) |


## 🔧 Edge Optimization: GridDensityBEV

We replaced the original `CQCA_cfa` module's CPU-based sklearn DBSCAN clustering with a fully GPU-native alternative called `GridDensityBEV`, designed for edge deployment on embedded platforms.

### Problem

The original CFA (Clustering Feature Assistance) module had a critical bottleneck for real-time and edge deployment:

1. **GPU→CPU data transfer** every forward pass to run sklearn DBSCAN
2. **Python-level for-loops** for density computation (O(n²))
3. **CPU→GPU transfer** to return results
4. **sklearn dependency** blocking ONNX/TensorRT export

### Solution: GridDensityBEV

All operations stay on GPU with no CPU transfer:

| Stage | Original (CQCA_cfa) | GridDensityBEV |
|:------|:---------------------|:---------------|
| Density estimation | sklearn DBSCAN (CPU) | `scatter_add` + `conv2d` with fixed kernel (GPU) |
| Noise filtering | DBSCAN labels + Python loop | Density thresholding via convolution |
| Cluster identity | DBSCAN label integers | Connected components via iterative `max_pool2d` |
| BEV map channels | velocity, raw density, cluster label | avg velocity, normalized density, cluster label |

**Connected Components on GPU**: Instead of losing cluster identity information, we recover it using iterative max-pooling label propagation. Each valid cell gets a unique position-based ID, then 3×3 `max_pool2d` is repeated for a fixed number of iterations until all cells in a connected component converge to the same label. This is fully ONNX/TensorRT compatible.

### BEV Map Output (3 channels → CNN → 64ch)

```
CH0: Average velocity (v_r_comp)     — improved over original last-write-wins
CH1: Normalized density              — neighbor count via 5×5 convolution
CH2: Cluster labels                  — GPU connected components (same semantics as DBSCAN)
```

### Impact on MAFF-Net Pipeline

The BEV map feeds into two downstream paths:

```
GridDensityBEV
  ├─→ spatial_features_img (B,64,H,W) → CQCA_caf Fuser (cross-attention with pillar BEV)
  │                                        → BACKBONE_2D → DENSE_HEAD → detections
  └─→ cluster_points (noise-filtered)  → PFE/CDA (proposal-centric keypoint sampling)
                                           → ROI_HEAD → refined detections
```

- **Fuser**: Cluster labels help cross-attention distinguish adjacent objects (e.g., two pedestrians side by side)
- **PFE**: Noise-filtered `cluster_points` improve keypoint sampling quality around proposals

### Properties

- **ONNX/TensorRT compatible**: No sklearn, numpy, or CPU dependencies
- **Deterministic**: Grid-based operations produce identical results every run
- **Differentiable**: Gradients can flow through the module (scatter_add + conv2d)
- **Module-level speedup**: ~10x faster than DBSCAN for the IMAGE_BACKBONE stage
- **Training speed**: Marginal improvement (~5-8%) since PFE and ROI_HEAD dominate epoch time
- **Inference latency**: Significant improvement for single-frame inference on edge devices

### Configuration

```yaml
IMAGE_BACKBONE:
    NAME: GridDensityBEV
    DBSCAN_MAP_W: 320
    DBSCAN_MAP_H: 320
    RESOLUTION: 0.16
    DBSCAN_EPS: 0.4          # controls density kernel size (5×5)
    DBSCAN_SAMPLE: 10        # min density threshold
    MAX_DENSITY: 100          # max density threshold
    CC_ITERATIONS: 20         # connected components iterations
```

To switch back to the original DBSCAN-based module, change `NAME: CQCA_cfa` in the config.

## 🙏 Acknowledgment

Many thanks to the open-source repositories:
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

## 📚 Citation

If you find our work valuable for your research, please consider citing our paper:


```shell

@ARTICLE{Bi_MAFF,
  author={Bi, Xin and Weng, Caien and Tong, Panpan and Fan, Baojie and Eichberge, Arno},
  journal={IEEE Robotics and Automation Letters}, 
  title={MAFF-Net: Enhancing 3D Object Detection With 4D Radar Via Multi-Assist Feature Fusion}, 
  year={2025},
  doi={10.1109/LRA.2025.3550707}}

```

