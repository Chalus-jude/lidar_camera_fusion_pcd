#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def rgb_guided_complete_depth(
    sparse_depth,
    valid_mask,
    guide_bgr,
    d=9,
    sigma_color=25.0,
    sigma_space=9.0,
):

    if sparse_depth.ndim != 2:
        raise ValueError(
            f"sparse_depth must be HxW, got shape {sparse_depth.shape}"
        )

    if valid_mask.shape != sparse_depth.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match "
            f"sparse_depth shape {sparse_depth.shape}"
        )

    if guide_bgr.ndim not in (2, 3):
        raise ValueError(
            f"guide image must be HxW or HxWx3, got shape {guide_bgr.shape}"
        )

    if guide_bgr.shape[:2] != sparse_depth.shape:
        raise ValueError(
            f"guide image shape {guide_bgr.shape[:2]} does not match "
            f"depth shape {sparse_depth.shape}"
        )


    if not valid_mask.any():
        return sparse_depth.copy()



    nn_idx = distance_transform_edt(
        ~valid_mask,
        return_distances=False,
        return_indices=True,
    )

    nn_filled = sparse_depth[tuple(nn_idx)].astype(np.float32)



    if guide_bgr.ndim == 3:
        guide_gray = cv2.cvtColor(
            guide_bgr,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        guide_gray = guide_bgr

    guide32 = guide_gray.astype(np.float32)
    depth32 = nn_filled.astype(np.float32)

    # ------------------------------------------------------------

    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "RGB-guided completion requires OpenCV contrib "
            "(cv2.ximgproc.jointBilateralFilter)."
        )


    refined = cv2.ximgproc.jointBilateralFilter(
        guide32,
        depth32,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    dense = refined.astype(np.float32)



    dense[valid_mask] = sparse_depth[valid_mask]

    return dense
