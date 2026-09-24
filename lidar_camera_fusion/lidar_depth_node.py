#!/usr/bin/env python3


import numpy as np
import cv2
import yaml
import time
import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import message_filters

from .lidar_depth_core import (
    project_lidar_to_image, points_to_sparse_depth, ip_basic_complete_depth,
)
from .lidar_depth_rgb_guided import rgb_guided_complete_depth


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    n_points = msg.width * msg.height

    if n_points == 0:
        return np.empty((0, 3), dtype=np.float64)

    field_offsets = {
        field.name: field.offset
        for field in msg.fields
    }

    required = ("x", "y", "z")

    if not all(name in field_offsets for name in required):
        raise RuntimeError(
            f"PointCloud2 is missing one or more XYZ fields. "
            f"Available fields: {list(field_offsets.keys())}"
        )

    if msg.is_bigendian:
        dtype = ">f4"
    else:
        dtype = "<f4"

    raw = np.frombuffer(
        msg.data,
        dtype=np.uint8
    ).reshape(n_points, msg.point_step)

    def read_float32_field(name):
        offset = field_offsets[name]

        return (
            raw[:, offset:offset + 4]
            .copy()
            .view(dtype)
            .reshape(-1)
        )

    x = read_float32_field("x")
    y = read_float32_field("y")
    z = read_float32_field("z")

    xyz = np.column_stack((x, y, z)).astype(np.float64)

    valid = np.all(np.isfinite(xyz), axis=1)

    return xyz[valid]


def load_calibration(path):
    with open(path, "r") as f:
        calib = yaml.safe_load(f)
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    D = np.array(calib["dist_coeffs"], dtype=np.float64)
    T_cam_lidar = np.array(calib["T_cam_lidar"], dtype=np.float64)
    return K, D, T_cam_lidar


def make_depth_image_msg(depth, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = depth.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
    return msg


class LidarToDepthNode(Node):
    def __init__(self):
        super().__init__("lidar_to_depth_node")

        self.declare_parameter("calibration_file", "")
        self.declare_parameter("lidar_topic", "/ouster/points")
        self.declare_parameter("min_range", 0.3)
        self.declare_parameter("max_range", 80.0)
        self.declare_parameter("publish_dense", True)
        self.declare_parameter("qos_reliability", "reliable")
        self.declare_parameter("image_width", 0)   # 0 = infer from calibration_file's resolution_wh if present
        self.declare_parameter("image_height", 0)
        self.declare_parameter( "camera_frame",
    "zedx_front_left_camera_frame_optical"
)
        self.declare_parameter("completion_downscale", 1.0)
        self.declare_parameter("log_timing", False)
        self.declare_parameter("rgb_topic", "")
        self.declare_parameter("rgb_slop", 0.05)
        self.declare_parameter("rgb_queue_size", 15)
        self.declare_parameter("bilateral_d", 9)
        self.declare_parameter("bilateral_sigma_color", 25.0)
        self.declare_parameter("bilateral_sigma_space", 9.0)
        self.completion_downscale = float(self.get_parameter("completion_downscale").value)
        if not (0.0 < self.completion_downscale <= 1.0):
            raise RuntimeError("completion_downscale must be in (0.0, 1.0]")

        calib_path = self.get_parameter("calibration_file").value
        if not calib_path:
            raise RuntimeError("calibration_file parameter is required")
        self.K, self.D, self.T_cam_lidar = load_calibration(calib_path)
        self.get_logger().info(f"Loaded calibration from {calib_path}")
        self.get_logger().info(f"K =\n{self.K}")
        self.get_logger().info(f"D = {self.D}")
        self.get_logger().info(f"T_cam_lidar =\n{self.T_cam_lidar}")

        with open(calib_path, "r") as f:
            raw_calib = yaml.safe_load(f)
        w = self.get_parameter("image_width").value
        h = self.get_parameter("image_height").value
        if w and h:
            self.img_wh = (w, h)
        elif "resolution_wh" in raw_calib:
            self.img_wh = tuple(int(x) for x in raw_calib["resolution_wh"])
        else:
            raise RuntimeError(
                "No image resolution: set image_width/image_height params, "
                "or add resolution_wh: [W, H] to the calibration file.")
        self.get_logger().info(f"image_wh = {self.img_wh}")

        if self.completion_downscale < 1.0:
            self.small_wh = (max(1, int(round(self.img_wh[0] * self.completion_downscale))),
                              max(1, int(round(self.img_wh[1] * self.completion_downscale))))
            self.get_logger().info(
                f"completion_downscale={self.completion_downscale} -> completing at "
                f"{self.small_wh} then upscaling back to {self.img_wh}")
        else:
            self.small_wh = self.img_wh

        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.camera_frame = self.get_parameter("camera_frame").value
        self.publish_dense = self.get_parameter("publish_dense").value

        self.log_timing = self.get_parameter("log_timing").value

        reliability = (ReliabilityPolicy.RELIABLE
                       if self.get_parameter("qos_reliability").value == "reliable"
                       else ReliabilityPolicy.BEST_EFFORT)
        qos = QoSProfile(depth=10, reliability=reliability,
                          durability=DurabilityPolicy.VOLATILE,
                          history=HistoryPolicy.KEEP_LAST)

        lidar_topic = self.get_parameter("lidar_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.bilateral_d = self.get_parameter("bilateral_d").value
        self.bilateral_sigma_color = self.get_parameter("bilateral_sigma_color").value
        self.bilateral_sigma_space = self.get_parameter("bilateral_sigma_space").value

        self.pub_sparse = self.create_publisher(Image, "/lidar_depth/sparse", 10)
        self.pub_dense = self.create_publisher(Image, "/lidar_depth/dense", 10) if self.publish_dense else None

        if self.rgb_topic:
            self.pub_rgb_fused = self.create_publisher(Image, "/lidar_depth/dense_rgb_fused", 10)
            lidar_sub = message_filters.Subscriber(self, PointCloud2, lidar_topic, qos_profile=qos)
            rgb_sub = message_filters.Subscriber(self, CompressedImage, self.rgb_topic, qos_profile=qos)
            slop = self.get_parameter("rgb_slop").value
            queue_size = self.get_parameter("rgb_queue_size").value
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [lidar_sub, rgb_sub], queue_size=queue_size, slop=slop)
            self.sync.registerCallback(self.on_cloud_and_image)
            self.get_logger().info(
                f"RGB fusion ENABLED: rgb_topic={self.rgb_topic} slop={slop} "
                f"queue_size={queue_size} -> publishing /lidar_depth/dense_rgb_fused "
                f"in addition to the LiDAR-only topics below.")
        else:
            self.pub_rgb_fused = None
            self.sub = self.create_subscription(PointCloud2, lidar_topic, self.on_cloud, qos)

        self.get_logger().info(
            f"lidar_to_depth_node ready. lidar_topic={lidar_topic} "
            f"min_range={self.min_range} max_range={self.max_range} "
            f"publish_dense={self.publish_dense} "
            f"completion_downscale={self.completion_downscale} "
            f"rgb_fusion={'ON (' + self.rgb_topic + ')' if self.rgb_topic else 'OFF'}")
        self.get_logger().info(  f"camera_frame={self.camera_frame}"
)

    def _compute_sparse_and_dense(self, msg: PointCloud2):

        t_start = time.time()
        xyz = pointcloud2_to_xyz(msg)
        if xyz.shape[0] == 0:
            return None

        u, v, z = project_lidar_to_image(
            xyz, self.T_cam_lidar, self.K, self.D, self.img_wh,
            self.min_range, self.max_range)
        sparse, valid_mask = points_to_sparse_depth(u, v, z, self.img_wh)
        t_sparse = time.time()

        dense = None
        if self.pub_dense is not None:
            if self.completion_downscale < 1.0:
                sparse_small = cv2.resize(sparse, self.small_wh, interpolation=cv2.INTER_NEAREST)
                mask_small = cv2.resize(valid_mask.astype(np.uint8), self.small_wh,
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                dense_small = ip_basic_complete_depth(sparse_small, mask_small, max_depth=self.max_range)
                dense = cv2.resize(dense_small, self.img_wh, interpolation=cv2.INTER_LINEAR)
                dense[valid_mask] = sparse[valid_mask]
            else:
                dense = ip_basic_complete_depth(sparse, valid_mask, max_depth=self.max_range)
        t_dense = time.time()

        if self.log_timing:
            self.get_logger().info(
                f"sparse={1000*(t_sparse-t_start):.1f}ms "
                f"dense={1000*(t_dense-t_sparse):.1f}ms "
                f"total={1000*(t_dense-t_start):.1f}ms")

        return sparse, valid_mask, dense

    def on_cloud(self, msg: PointCloud2):
        try:
            result = self._compute_sparse_and_dense(msg)
            if result is None:
                return
            sparse, valid_mask, dense = result
            frame_id = self.camera_frame
            self.pub_sparse.publish(make_depth_image_msg(sparse, frame_id, msg.header.stamp))
            if dense is not None:
                self.pub_dense.publish(make_depth_image_msg(dense, frame_id, msg.header.stamp))
        except Exception:
            self.get_logger().error("Exception in on_cloud callback:\n" + traceback.format_exc())

    def on_cloud_and_image(self, cloud_msg: PointCloud2, image_msg: CompressedImage):
        try:
            result = self._compute_sparse_and_dense(cloud_msg)
            if result is None:
                return
            sparse, valid_mask, dense = result
            frame_id = self.camera_frame
            self.pub_sparse.publish(make_depth_image_msg(sparse, frame_id, cloud_msg.header.stamp))
            if dense is not None:
                self.pub_dense.publish(make_depth_image_msg(dense, frame_id, cloud_msg.header.stamp))

            t0 = time.time()
            guide_bgr = cv2.imdecode(np.frombuffer(image_msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if guide_bgr is None:
                self.get_logger().warn("Failed to decode CompressedImage for RGB fusion; skipping this frame.")
                return
            if guide_bgr.shape[1::-1] != self.img_wh:
                guide_bgr = cv2.resize(guide_bgr, self.img_wh, interpolation=cv2.INTER_LINEAR)
            fused = rgb_guided_complete_depth(
                sparse, valid_mask, guide_bgr,
                d=self.bilateral_d, sigma_color=self.bilateral_sigma_color,
                sigma_space=self.bilateral_sigma_space)
            self.pub_rgb_fused.publish(make_depth_image_msg(fused, frame_id, cloud_msg.header.stamp))
            if self.log_timing:
                self.get_logger().info(f"rgb_fused={1000*(time.time()-t0):.1f}ms")
        except Exception:
            self.get_logger().error("Exception in on_cloud_and_image callback:\n" + traceback.format_exc())


def main():
    rclpy.init()
    node = LidarToDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
