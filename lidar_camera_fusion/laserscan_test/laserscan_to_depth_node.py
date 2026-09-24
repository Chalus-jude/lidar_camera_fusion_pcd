 #!/usr/bin/env python3


import numpy as np
import cv2
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Image

from lidar_depth_core import (
    project_lidar_to_image, points_to_sparse_depth, ip_basic_complete_depth,
)


def laserscan_to_xyz(scan: LaserScan, min_range=None, max_range=None) -> np.ndarray:

    ranges = np.asarray(scan.ranges, dtype=np.float64)
    n = ranges.shape[0]
    angles = scan.angle_min + np.arange(n) * scan.angle_increment

    lo = scan.range_min if min_range is None else max(scan.range_min, min_range)
    hi = scan.range_max if max_range is None else min(scan.range_max, max_range)
    valid = np.isfinite(ranges) & (ranges >= lo) & (ranges <= hi)
    ranges, angles = ranges[valid], angles[valid]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def load_calibration(path):
    with open(path, "r") as f:
        calib = yaml.safe_load(f)
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    D = np.array(calib["dist_coeffs"], dtype=np.float64)
    T_cam_lidar = np.array(calib["T_cam_lidar"], dtype=np.float64)
    return K, D, T_cam_lidar, calib


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


class LaserScanToDepthNode(Node):
    def __init__(self):
        super().__init__("laserscan_to_depth_node")

        self.declare_parameter("calibration_file", "")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("min_range", 0.0)   # 0.0 = use the LaserScan msg's own range_min
        self.declare_parameter("max_range", 0.0)   # 0.0 = use the LaserScan msg's own range_max
        self.declare_parameter("publish_dense", False)
        self.declare_parameter("qos_reliability", "reliable")
        self.declare_parameter("image_width", 0)
        self.declare_parameter("image_height", 0)

        calib_path = self.get_parameter("calibration_file").value
        if not calib_path:
            raise RuntimeError("calibration_file parameter is required")
        self.K, self.D, self.T_cam_lidar, raw_calib = load_calibration(calib_path)
        self.get_logger().info(f"Loaded calibration from {calib_path}")
        self.get_logger().info(f"K =\n{self.K}")
        self.get_logger().info(f"D = {self.D}")
        self.get_logger().info(f"T_cam_lidar =\n{self.T_cam_lidar}")

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

        min_range_param = self.get_parameter("min_range").value
        max_range_param = self.get_parameter("max_range").value
        self.min_range = min_range_param if min_range_param > 0.0 else None
        self.max_range = max_range_param if max_range_param > 0.0 else None
        self.publish_dense = self.get_parameter("publish_dense").value

        if self.publish_dense:
            self.get_logger().warn(
                "publish_dense=true on a 2D LaserScan: /laser_depth/dense_CAUTION "
                "will vertically smear a single scan-height band's values across "
                "the whole image. This is NOT a real per-pixel depth estimate -- "
                "see this file's module docstring before using it downstream.")

        reliability = (ReliabilityPolicy.RELIABLE
                       if self.get_parameter("qos_reliability").value == "reliable"
                       else ReliabilityPolicy.BEST_EFFORT)
        qos = QoSProfile(depth=10, reliability=reliability,
                          durability=DurabilityPolicy.VOLATILE,
                          history=HistoryPolicy.KEEP_LAST)

        scan_topic = self.get_parameter("scan_topic").value
        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, qos)

        self.pub_sparse = self.create_publisher(Image, "/laser_depth/sparse", 10)
        self.pub_dense = (self.create_publisher(Image, "/laser_depth/dense_CAUTION", 10)
                           if self.publish_dense else None)

        self.get_logger().info(
            f"laserscan_to_depth_node ready. scan_topic={scan_topic} "
            f"publish_dense={self.publish_dense}")

    def on_scan(self, msg: LaserScan):
        try:
            xyz = laserscan_to_xyz(msg, self.min_range, self.max_range)
            if xyz.shape[0] == 0:
                return

            u, v, z = project_lidar_to_image(
                xyz, self.T_cam_lidar, self.K, self.D, self.img_wh,
                self.min_range or msg.range_min, self.max_range or msg.range_max)
            sparse, valid_mask = points_to_sparse_depth(u, v, z, self.img_wh)

            frame_id = "camera_color_optical_frame"
            self.pub_sparse.publish(make_depth_image_msg(sparse, frame_id, msg.header.stamp))

            if self.pub_dense is not None:
                dense = ip_basic_complete_depth(
                    sparse, valid_mask, max_depth=(self.max_range or msg.range_max))
                self.pub_dense.publish(make_depth_image_msg(dense, frame_id, msg.header.stamp))
        except Exception:
            import traceback
            self.get_logger().error("Exception in on_scan callback:\n" + traceback.format_exc())


def main():
    rclpy.init()
    node = LaserScanToDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
