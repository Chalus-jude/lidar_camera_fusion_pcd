#R1- transformed code since it is /scan and not pcd

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def project_lidar_to_image(points_lidar, T_cam_lidar, K, D, img_wh,
                            min_range, max_range, range_mode="camera_z"):

    W, H = img_wh
    n = points_lidar.shape[0]
    empty = (np.empty(0), np.empty(0), np.empty(0))
    if n == 0:
        return empty

    homog = np.hstack([points_lidar, np.ones((n, 1))])
    points_cam = (T_cam_lidar @ homog.T).T[:, :3]
    z = points_cam[:, 2]

    if range_mode == "euclidean":
        range_val = np.linalg.norm(points_cam, axis=1)
    elif range_mode == "camera_z":
        range_val = z
    else:
        raise ValueError(f"range_mode must be 'camera_z' or 'euclidean', got {range_mode!r}")

    in_range = (range_val > min_range) & (range_val < max_range) & (z > 0)
    points_cam, z = points_cam[in_range], z[in_range]
    if points_cam.shape[0] == 0:
        return empty

    uv, _ = cv2.projectPoints(points_cam, np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return u[in_bounds], v[in_bounds], z[in_bounds]


def points_to_sparse_depth(u, v, z, img_wh):
    """Nearest-point-wins z-buffer rasterization -> (depth_map, valid_mask)."""
    W, H = img_wh
    depth = np.full((H, W), np.inf, dtype=np.float32)
    if u.size == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, zc = ui[in_bounds], vi[in_bounds], z[in_bounds]
    order = np.argsort(-zc)  # process far-to-near; min() below keeps nearest regardless of order
    ui, vi, zc = ui[order], vi[order], zc[order]
    depth[vi, ui] = np.minimum(depth[vi, ui], zc)
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return depth, mask


_DIAMOND_KERNEL_5 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=np.uint8)


def _full_kernel(k):
    return np.ones((k, k), dtype=np.uint8)


def ip_basic_complete_depth(sparse_depth, valid_mask, max_depth=80.0,
                             close_kernel_size=5, small_fill_kernel_size=7,
                             large_fill_kernel_size=15,
                             extrapolate_top=False, guarantee_fill=True,
                             blur_type="gaussian"):

    if valid_mask.any():
        measured_max = sparse_depth[valid_mask].max()
        if measured_max > max_depth:
            raise ValueError(
                f"max_depth={max_depth} is less than the max measured depth "
                f"in sparse_depth ({measured_max:.3f}). max_depth must be >= "
                f"the max_range used to build this sparse_depth.")

    depth = sparse_depth.copy()
    depth[~valid_mask] = 0.0

    inverted = np.where(depth > 0, max_depth - depth, 0.0).astype(np.float32)

    dilated = cv2.dilate(inverted, _DIAMOND_KERNEL_5)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, _full_kernel(close_kernel_size))

    filled = closed.copy()
    empty_mask = filled == 0
    dilated_small = cv2.dilate(filled, _full_kernel(small_fill_kernel_size))
    filled[empty_mask] = dilated_small[empty_mask]

    empty_mask = filled == 0
    dilated_large = cv2.dilate(filled, _full_kernel(large_fill_kernel_size))
    filled[empty_mask] = dilated_large[empty_mask]

    if extrapolate_top:
        h, w = filled.shape
        for col in range(w):
            col_vals = filled[:, col]
            nonzero = np.nonzero(col_vals)[0]
            if nonzero.size > 0:
                top_row = nonzero[0]
                filled[:top_row, col] = col_vals[top_row]

    if guarantee_fill:
        still_empty = filled == 0
        if still_empty.any() and (~still_empty).any():
            nn_idx = distance_transform_edt(still_empty, return_distances=False, return_indices=True)
            nn_filled = filled[tuple(nn_idx)]
            filled[still_empty] = nn_filled[still_empty]

    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(filled, (5, 5), 0)
        filled[filled > 0] = blurred[filled > 0]
    elif blur_type == "median":
        blurred = cv2.medianBlur(filled.astype(np.float32), 5)
        filled[filled > 0] = blurred[filled > 0]

    dense = np.where(filled > 0, max_depth - filled, 0.0).astype(np.float32)


    dense[valid_mask] = sparse_depth[valid_mask]

    return dense
