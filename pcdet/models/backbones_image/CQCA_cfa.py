# ！/usr/bin/python3
# _*_coding: utf-8 _*_
#
# Copyright (C) 2024 - 2024 Caien Weng, Inc. All Rights Reserved
#
# @Time   : 2024/4/18 下午7:53
# @Author : Caien Weng
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class CQCA_cfa(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dbscan_map_w = self.model_cfg.DBSCAN_MAP_W
        self.dbscan_map_h = self.model_cfg.DBSCAN_MAP_H
        self.dbscan_feature = self.model_cfg.DBSCAN_FEATURE
        self.dbscan_v = self.model_cfg.DBSCAN_V
        self.dbscan_y = self.model_cfg.DBSCAN_Y
        self.resolution = self.model_cfg.RESOLUTION
        self.dbscan_eps = self.model_cfg.DBSCAN_EPS
        self.dbscan_sample = self.model_cfg.DBSCAN_SAMPLE
        self.point_x = self.model_cfg.POINTX
        self.point_y = self.model_cfg.POINTY
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def generate_map(self, points_xyv, eps, samples):
        points_xyv = points_xyv.cpu()
        dbscan = DBSCAN(eps=eps, min_samples=samples)
        labels = torch.tensor(dbscan.fit_predict(points_xyv))
        density_values = torch.zeros_like(labels, dtype=torch.float)
        for i, label in enumerate(labels):
            if label != -1:
                neighborhood_indices = torch.where(labels == label)[0]
                density_values[i] = len(neighborhood_indices)

        for i, density in enumerate(density_values):
            if density > 100 and labels[i] != -1:
                labels[i] = -1  # 标记为噪声
        points_density = torch.cat([points_xyv, density_values.reshape(-1, 1), labels.reshape(-1, 1)], dim=1)
        dbscan_points = points_density[torch.where(labels > 0)]
        # 初始化 BEV 地图
        resolution = self.resolution
        bev_map = torch.zeros((self.dbscan_map_w, self.dbscan_map_h, self.dbscan_feature),
                              dtype=torch.float32).to(points_xyv.device)
        # 计算点云在 BEV 地图中的位置
        bev_x = ((dbscan_points[:, 0]) / resolution).long() - 1
        bev_y = ((dbscan_points[:, 1] + self.dbscan_y) / resolution).long() - 1
        bev_map[bev_y, bev_x] = dbscan_points[:, 2:]
        return bev_map.permute(2, 0, 1).cuda(), labels

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']
        batch_size = batch_dict['batch_size']
        dbscan_map2 = []
        cluster_points = []
        for batch_idx in range(batch_size):
            batch_mask = points[:, 0] == batch_idx
            point = points[batch_mask]
            point_xyv = torch.cat([point[:, 1:3], point[:, self.dbscan_v].reshape(-1, 1)], dim=1)
            db_map, db_labels = self.generate_map(point_xyv, self.dbscan_eps, self.dbscan_sample)
            dbscan_map2.append(db_map)
            final_points = point[db_labels != -1]
            cluster_points.append(final_points)

        dbscan_map2 = torch.stack(dbscan_map2, 0)
        cluster_points = torch.cat(cluster_points, 0)
        dbscan_map2 = dbscan_map2.view(batch_size, 3, self.dbscan_map_w, self.dbscan_map_h)
        out = self.conv(dbscan_map2)
        batch_dict['spatial_features_img'] = out
        batch_dict['cluster_points'] = cluster_points
        return batch_dict


def show_result(kde_map_feature):
    import matplotlib.pyplot as plt
    import numpy as np
    intermediate_image = kde_map_feature[0].cpu().detach().numpy()
    # 将多个通道相加
    summed_image = np.sum(intermediate_image, axis=0)  # 沿着通道维度求和
    plt.figure(figsize=(10, 5))
    plt.imshow(summed_image, cmap='gray')
    plt.title('Density Image')
    plt.show()


class AdaptiveDBSCAN:
    def __init__(self, min_samples=5):
        self.min_samples = min_samples

    def fit(self, points):
        self.points = points
        self.X = points[:, 0:2]
        X = points[:, 0:2]
        self.labels = np.zeros(len(X))
        self.cluster_idx = 0
        self.kd_tree = KDTree(X)
        self.alte = 2.5
        for i in range(len(X)):
            if self.labels[i] == 0:  # unvisited point
                if self.expand_cluster(i):
                    self.cluster_idx += 1

    def expand_cluster(self, idx):
        neighbors = self.query_neighbors(idx)
        if len(neighbors) < self.min_samples:
            self.labels[idx] = -1  # noise point
            return False
        else:
            self.labels[idx] = self.cluster_idx
            for neighbor_idx in neighbors:
                if self.labels[neighbor_idx] == 0:  # unvisited point
                    self.labels[neighbor_idx] = self.cluster_idx
                    neighbor_neighbors = self.query_neighbors(neighbor_idx)
                    if len(neighbor_neighbors) >= self.min_samples:
                        neighbors = np.append(neighbors, neighbor_neighbors)
            return True

    def query_neighbors(self, idx):
        eps = self.alte * np.linalg.norm(self.X[idx]) * np.tan(np.pi / 180 * 0.75)
        if eps < self.alte * 0.2:
            eps = self.alte * 0.2
        neighbors = self.kd_tree.query_radius([self.X[idx]], r=eps)[0]
        return neighbors


class GridDensityBEV(nn.Module):
    """
    DBSCAN yerine GPU-native grid-based density estimation + connected components.

    Ayni ciktilari uretir:
      - batch_dict['spatial_features_img']: (B, 64, H, W) BEV feature map
      - batch_dict['cluster_points']: (N_filtered, C) gurultu olmayan noktalar

    CH0: average velocity  (orijinal: last-write velocity)
    CH1: normalized density (orijinal: raw density)
    CH2: cluster labels     (orijinal ile ayni semantik - connected components)

    Tum islemler GPU'da kalir, CPU transferi yok.
    ONNX/TensorRT uyumlu (F.max_pool2d + sabit iterasyon).
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_h = self.model_cfg.DBSCAN_MAP_W   # 320
        self.grid_w = self.model_cfg.DBSCAN_MAP_H   # 320
        self.resolution = self.model_cfg.RESOLUTION  # 0.16
        self.velocity_idx = self.model_cfg.DBSCAN_V  # 6 (v_r_comp column in points)
        self.y_offset = self.model_cfg.DBSCAN_Y      # 25.6

        # DBSCAN eps=0.4, resolution=0.16 -> 0.4/0.16 = 2.5 cells -> 5x5 kernel
        eps = self.model_cfg.DBSCAN_EPS
        kernel_size = max(3, int(2 * round(eps / self.resolution) + 1))  # 5
        self.density_threshold = self.model_cfg.DBSCAN_SAMPLE  # 10 (= min_samples)
        self.max_density = self.model_cfg.get('MAX_DENSITY', 100)

        # Connected components iterations (radar clusters are small, 20 is sufficient)
        self.cc_iterations = self.model_cfg.get('CC_ITERATIONS', 20)

        # Fixed density counting kernel (not learnable)
        self.register_buffer(
            'density_kernel',
            torch.ones(1, 1, kernel_size, kernel_size)
        )
        self.density_padding = kernel_size // 2  # 2 for k=5

        # Same CNN as CQCA_cfa: 3 -> 16 -> 32 -> 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def _connected_components_gpu(self, valid_mask):
        """
        GPU-based connected components via iterative max-pooling.

        valid_mask: (H, W) bool tensor
        Returns: (H, W) float tensor with normalized component labels [0, 1]

        Mantik: Her gecerli hucreye benzersiz ID ver (pozisyon bazli),
        sonra 3x3 max-pool ile ayni bilesendeki tum hucreler en buyuk
        ID'ye yakinsar. Radar clusterlari kucuk oldugundan 20 iterasyon yeter.
        """
        H, W = valid_mask.shape
        device = valid_mask.device

        # Each valid cell gets unique ID: row * W + col + 1 (avoid 0)
        ids = torch.arange(1, H * W + 1, device=device, dtype=torch.float32).view(H, W)
        labels = ids * valid_mask.float()  # 0 for invalid cells

        # Iteratively propagate max label to 3x3 neighbors
        for _ in range(self.cc_iterations):
            labels_4d = labels.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            pooled = torch.nn.functional.max_pool2d(
                labels_4d, kernel_size=3, stride=1, padding=1
            ).squeeze(0).squeeze(0)  # (H, W)
            # Only keep labels for valid cells
            labels = pooled * valid_mask.float()

        # Normalize to [0, 1] for CNN input
        label_max = labels.max().clamp(min=1)
        return labels / label_max

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (N, 8): [batch_idx, x, y, z, rcs, v_r, v_r_comp, time]
        batch_size = batch_dict['batch_size']
        device = points.device

        bev_maps = []
        cluster_points_list = []

        for batch_idx in range(batch_size):
            batch_mask = points[:, 0] == batch_idx
            point = points[batch_mask]  # (n, 8)

            if point.shape[0] == 0:
                bev_maps.append(torch.zeros(3, self.grid_h, self.grid_w, device=device))
                cluster_points_list.append(point[:0])
                continue

            # --- Step 1: Compute grid indices for each point ---
            gx = (point[:, 1] / self.resolution).long().clamp(0, self.grid_w - 1)
            gy = ((point[:, 2] + self.y_offset) / self.resolution).long().clamp(0, self.grid_h - 1)
            velocity = point[:, self.velocity_idx]

            # --- Step 2: Scatter points into BEV grid ---
            linear_idx = gy * self.grid_w + gx  # (n,)

            grid_size = self.grid_h * self.grid_w
            count_flat = torch.zeros(grid_size, device=device)
            vel_sum_flat = torch.zeros(grid_size, device=device)

            count_flat.scatter_add_(0, linear_idx, torch.ones_like(velocity))
            vel_sum_flat.scatter_add_(0, linear_idx, velocity)

            # Reshape to 2D grids
            count_grid = count_flat.view(1, 1, self.grid_h, self.grid_w)
            vel_sum_grid = vel_sum_flat.view(self.grid_h, self.grid_w)

            # --- Step 3: Density map via convolution (neighbor counting) ---
            density_map = torch.nn.functional.conv2d(
                count_grid, self.density_kernel, padding=self.density_padding
            ).squeeze(0).squeeze(0)  # (H, W)

            # --- Step 4: Noise mask (like DBSCAN min_samples + max density) ---
            noise_mask = (density_map < self.density_threshold) | (density_map > self.max_density)
            valid_mask = ~noise_mask  # cells that belong to clusters

            # Average velocity (avoid division by zero)
            safe_count = count_flat.view(self.grid_h, self.grid_w).clamp(min=1)
            vel_mean_grid = vel_sum_grid / safe_count

            # Zero out noise regions
            vel_mean_grid[noise_mask] = 0
            density_clean = density_map.clone()
            density_clean[noise_mask] = 0

            # --- Step 5: Connected components for cluster labels ---
            cluster_label_map = self._connected_components_gpu(valid_mask)

            # --- Step 6: Normalize and stack 3-channel BEV map ---
            density_max = density_clean.max().clamp(min=1)

            bev_map = torch.stack([
                vel_mean_grid,                  # CH0: average velocity
                density_clean / density_max,    # CH1: normalized density
                cluster_label_map,              # CH2: cluster labels (connected components)
            ], dim=0)  # (3, H, W)

            bev_maps.append(bev_map)

            # --- Step 7: Filter cluster_points (like DBSCAN noise removal) ---
            point_density = density_map[gy, gx]
            valid_point_mask = (point_density >= self.density_threshold) & \
                               (point_density <= self.max_density)
            cluster_points_list.append(point[valid_point_mask])

        # Stack batch
        bev_maps = torch.stack(bev_maps, dim=0)  # (B, 3, H, W)
        cluster_points = torch.cat(cluster_points_list, dim=0)

        # Same CNN as original CQCA_cfa
        out = self.conv(bev_maps)

        batch_dict['spatial_features_img'] = out       # (B, 64, 320, 320)
        batch_dict['cluster_points'] = cluster_points  # (N_filtered, 8)
        return batch_dict
