# LiDAR Camera Fusion

A ROS 2 perception pipeline for generating **dense camera-frame depth from 3D LiDAR data**, with optional camera-guided refinement.

The project focuses on transforming sparse LiDAR measurements into a camera-aligned depth representation that can be used for perception, visualization, and downstream robotic applications.

## Overview

A 3D LiDAR provides accurate geometric measurements but produces sparse depth when projected onto an image plane. A camera provides dense visual information but does not necessarily provide reliable depth at every pixel.

This project explores their combination:

```text
3D LiDAR
   ↓
Point Cloud Processing
   ↓
Camera Projection
   ↓
Sparse Depth
   ↓
Depth Completion
   ↓
Dense LiDAR Depth
   ↓
Optional RGB Guidance
   ↓
Fused Camera Depth
```

## Pipeline

The system takes:

* 3D LiDAR point clouds
* Camera calibration parameters
* LiDAR-to-camera extrinsic transformation
* Optional RGB images
* Optional reference depth

The LiDAR points are transformed into the camera coordinate frame and projected onto the image plane.

The resulting sparse depth map is then processed to generate a denser depth representation.

## Processing Modes

### LiDAR-only

Uses only LiDAR geometry to generate dense depth.

```text
LiDAR
  ↓
Projection
  ↓
Sparse Depth
  ↓
Depth Completion
  ↓
Dense Depth
```

This provides a geometry-driven baseline without relying on RGB information.

### RGB-guided Fusion

Camera information can optionally be used to refine the LiDAR-derived depth.

```text
LiDAR ──────→ Sparse Depth ──┐
                             ├──→ RGB-guided Fusion
RGB Image ───────────────────┘
                             ↓
                       Fused Depth
```

This allows image structure to guide depth propagation while retaining LiDAR measurements as the geometric reference.

## ROS 2

The pipeline is implemented as ROS 2 nodes and operates on standard sensor topics.

Typical inputs include:

```text
/points
/camera/image
/camera/camera_info
```

The system publishes depth representations such as:

```text
/lidar_depth/sparse
/lidar_depth/dense
/lidar_depth/dense_rgb_fused
```

The generated depth is expressed in the camera frame for direct visualization and downstream processing.

## Calibration

Accurate fusion depends on both intrinsic and extrinsic calibration.

The pipeline uses:

* Camera intrinsic matrix
* Camera distortion parameters
* LiDAR-to-camera translation
* LiDAR-to-camera rotation
* Camera image resolution

The calibration is applied during LiDAR point projection to establish pixel-level correspondence between the sensors.

## Evaluation

When reference depth is available, the generated depth can be evaluated using metrics such as:

* Valid depth coverage
* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Relative depth error
* Pixel-wise depth comparison

The project also supports visual comparison between the generated depth and reference camera depth.

## Applications

The resulting camera-aligned depth can support:

* Robot perception
* Obstacle detection
* Depth-aware navigation
* Costmap generation
* Sensor fusion
* 3D scene understanding
* Depth-based object perception

## Technology Stack

* ROS 2 Humble
* Python
* OpenCV
* NumPy
* Point Cloud Processing
* 3D LiDAR
* RGB / Depth Cameras
* RViz 2

## Status

**Active Development**

Current work focuses on:

* LiDAR-to-camera projection
* Sparse depth generation
* LiDAR-only depth completion
* RGB-guided depth fusion
* Calibration validation
* Quantitative depth evaluation
* ROS 2 integration

## Focus

**LiDAR-Camera Fusion · 3D Perception · Depth Estimation · Sensor Fusion · ROS 2 · Computer Vision · Robotics**
