#!/usr/bin/env python3

import os

import numpy as np
import yaml
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

import message_filters
from sensor_msgs.msg import PointCloud2, CompressedImage, Image, PointField
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

from scipy.ndimage import distance_transform_edt


def load_calibration(path):

    with open(path, 'r') as f:
        calib = yaml.safe_load(f)

    if 'camera_matrix' not in calib:
        raise KeyError(f"'camera_matrix' missing from calibration file {path}")
    if 'T_cam_lidar' not in calib:
        raise KeyError(f"'T_cam_lidar' missing from calibration file {path}")

    K = np.array(calib['camera_matrix'], dtype=np.float64).reshape(3, 3)

    D = np.array(
        calib.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(-1)

    P = calib.get('projection_matrix', None)
    if P is not None:
        P = np.array(P, dtype=np.float64).reshape(3, 4)

    T = np.array(calib['T_cam_lidar'], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            f"T_cam_lidar must be a 4x4 matrix, got shape {T.shape} in {path}"
        )

    return K, D, P, T


class LidarCameraFusionNode(Node):

    def __init__(self):
        super().__init__('lidar_camera_fusion_node')

        self._declare_params()
        self._read_params()
        self._load_calibration()
        self._setup_io()

        self._frame_count = 0
        self.get_logger().info(
            f'lidar_camera_fusion_node ready. lidar_topic={self.lidar_topic} '
            f'image_topic={self.image_topic} slop={self.slop} '
            f'queue_size={self.queue_size} image_is_rectified={self.image_is_rectified}'
        )


    def _declare_params(self):
        self.declare_parameter('lidar_topic', '/ouster/points')
        self.declare_parameter(
            'image_topic',
            '/zedx_front/zed_node/rgb/color/rect/image/compressed',
        )
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('image_is_rectified', True)

        self.declare_parameter('queue_size', 10)
        self.declare_parameter('slop', 0.03)

        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 60.0)
        self.declare_parameter('depth_bias_m', 0.0)

        self.declare_parameter('publish_sparse_depth', True) # outpt publish bool for spare depth before densification
        self.declare_parameter('publish_dense_depth', True)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('publish_colorized_cloud', False)

        self.declare_parameter('qos_reliability', 'reliable')

        self.declare_parameter('bilateral_d', 9) #depth smoothening and controls edge bleeding
        self.declare_parameter('bilateral_sigma_color', 25.0) #controlling greyscale intensification wrt rgb edges; the lower the stricter the rgb edge
        self.declare_parameter('bilateral_sigma_space', 9.0) # hole densification

        self.declare_parameter('log_every_n_frames', 30)

    def _read_params(self):
        gp = self.get_parameter
        self.lidar_topic = gp('lidar_topic').value
        self.image_topic = gp('image_topic').value
        self.calib_path = gp('calibration_file').value
        self.image_is_rectified = gp('image_is_rectified').value

        self.queue_size = gp('queue_size').value
        self.slop = gp('slop').value

        self.min_range = gp('min_range').value
        self.max_range = gp('max_range').value
        self.depth_bias_m = gp('depth_bias_m').value

        self.do_sparse = gp('publish_sparse_depth').value
        self.do_dense = gp('publish_dense_depth').value
        self.do_overlay = gp('publish_overlay').value
        self.do_cloud = gp('publish_colorized_cloud').value

        self.qos_reliability = gp('qos_reliability').value

        self.bilateral_d = gp('bilateral_d').value
        self.bilateral_sigma_color = gp('bilateral_sigma_color').value
        self.bilateral_sigma_space = gp('bilateral_sigma_space').value

        self.log_every_n = gp('log_every_n_frames').value

    def _load_calibration(self):
        if not self.calib_path:
            raise RuntimeError(
                "Parameter 'calibration_file' is not set. Provide a path "
                "to a calibration YAML with camera_matrix / dist_coeffs / "
                "projection_matrix (optional) / T_cam_lidar."
            )
        if not os.path.isfile(self.calib_path):
            raise FileNotFoundError(f'Calibration file not found: {self.calib_path}')

        self.K, self.D, self.P, self.T_cam_lidar = load_calibration(self.calib_path)

        self.get_logger().info(f'Loaded calibration from {self.calib_path}')
        self.get_logger().info(f'K =\n{self.K}')
        self.get_logger().info(f'D = {self.D}')
        self.get_logger().info(    f'depth_bias_m={self.depth_bias_m:.3f}')

        self.get_logger().info(
            f'P =\n{self.P}' if self.P is not None else 'P = <not provided>'
        )
        self.get_logger().info(f'T_cam_lidar =\n{self.T_cam_lidar}')

        if self.image_is_rectified and self.P is None:
            self.get_logger().warn(
                "image_is_rectified=True but no 'projection_matrix' (P) was "
                "given in the calibration file. Falling back to K with zero "
                "distortion. This is a placeholder -- extract the real P "
                "from /zedx_front/zed_node/rgb/color/rect/image/camera_info "
                "and add it to the calibration YAML before trusting output."
            )

        if self.image_is_rectified:
            self._proj_K = self.P[:, :3] if self.P is not None else self.K
            self._proj_D = np.zeros(5)
        else:
            self._proj_K = self.K
            self._proj_D = self.D

    def _setup_io(self):
        self.bridge = CvBridge()

        reliability = (
            ReliabilityPolicy.RELIABLE
            if self.qos_reliability == 'reliable'
            else ReliabilityPolicy.BEST_EFFORT
        )
        sensor_qos = QoSProfile(
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.queue_size,
        )

        self.lidar_sub = message_filters.Subscriber(
            self, PointCloud2, self.lidar_topic, qos_profile=sensor_qos
        )
        self.image_sub = message_filters.Subscriber(
            self, CompressedImage, self.image_topic, qos_profile=sensor_qos
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.image_sub],
            queue_size=self.queue_size,
            slop=self.slop,
        )
        self.sync.registerCallback(self.fusion_callback)

        pub_qos = QoSProfile(depth=10)
        if self.do_sparse:
            self.sparse_pub = self.create_publisher(Image, '/fusion/sparse_depth', pub_qos)
        if self.do_dense:
            self.dense_pub = self.create_publisher(Image, '/fusion/dense_depth', pub_qos)
        if self.do_overlay:
            self.overlay_pub = self.create_publisher(Image, '/fusion/overlay', pub_qos)
        if self.do_cloud:
            self.cloud_pub = self.create_publisher(PointCloud2, '/fusion/colorized_cloud', pub_qos)


    def pointcloud2_to_xyz(self, msg):
        pts = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        arr = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float64) \
            if hasattr(pts, 'dtype') and pts.dtype.names else \
            np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
        return arr

    def project_to_image(self, points_lidar, img_shape):
        H, W = img_shape[:2]
        n = points_lidar.shape[0]
        empty = (np.empty(0), np.empty(0), np.empty(0))
        if n == 0:
            return empty

        homog = np.hstack([points_lidar, np.ones((n, 1))])
        points_cam = (self.T_cam_lidar @ homog.T).T[:, :3]

        z = points_cam[:, 2]
        in_range = (z > self.min_range) & (z < self.max_range)
        points_cam = points_cam[in_range]
        z = z[in_range]
        if points_cam.shape[0] == 0:
            return empty

        rvec = np.zeros(3)
        tvec = np.zeros(3)
        uv, _ = cv2.projectPoints(points_cam, rvec, tvec, self._proj_K, self._proj_D)
        uv = uv.reshape(-1, 2)
        u, v = uv[:, 0], uv[:, 1]

        in_bounds = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        return u[in_bounds], v[in_bounds], z[in_bounds]

    def build_sparse_depth(self, u, v, z, img_shape):
        H, W = img_shape[:2]
        depth = np.full((H, W), np.inf, dtype=np.float32)

        if u.size == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        order = np.argsort(-z)
        ui, vi, zs = ui[order], vi[order], z[order]
        depth[vi, ui] = np.minimum(depth[vi, ui], zs)

        mask = np.isfinite(depth)
        depth[~mask] = 0.0
        return depth, mask

    def densify(self, sparse_depth, valid_mask, guide_bgr):
        if not valid_mask.any():
            return sparse_depth.copy()

        idx = distance_transform_edt(
            ~valid_mask,
            return_distances=False,
            return_indices=True
        )
        nn_filled = sparse_depth[tuple(idx)].astype(np.float32)

        # Convert RGB/BGR guidance image to grayscale.
        if guide_bgr.ndim == 3:
            guide_gray = cv2.cvtColor(
                guide_bgr,
                cv2.COLOR_BGR2GRAY
            )
        else:
            guide_gray = guide_bgr


        guide32 = guide_gray.astype(np.float32)
        depth32 = nn_filled.astype(np.float32)

        if hasattr(cv2, 'ximgproc'):
            refined = cv2.ximgproc.jointBilateralFilter(
                guide32,
                depth32,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
        else:
            refined = cv2.bilateralFilter(
                depth32,
                d=self.bilateral_d,
                sigmaColor=max(
                    self.bilateral_sigma_color * 0.02,
                    1e-3
                ),
                sigmaSpace=self.bilateral_sigma_space,
            )

        refined[valid_mask] = sparse_depth[valid_mask]

        return refined.astype(np.float32)

    def make_overlay(self, bgr_image, u, v, z):
        overlay = bgr_image.copy()
        if u.size == 0:
            return overlay
        span = max(self.max_range - self.min_range, 1e-6)
        z_norm = np.clip((z - self.min_range) / span, 0, 1)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for i in range(u.size):
            cv2.circle(
                overlay,
                (int(round(u[i])), int(round(v[i]))),
                1,
                tuple(int(c) for c in colors[i, 0]),
                -1,
            )
        return overlay

    def backproject_colorized_cloud(self, dense_depth, bgr_image, header):
        H, W = dense_depth.shape
        fx, fy = self._proj_K[0, 0], self._proj_K[1, 1]
        cx, cy = self._proj_K[0, 2], self._proj_K[1, 2]

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        z = dense_depth
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.uint8)

        valid = points[:, 2] > 0
        points = points[valid]
        rgb = rgb[valid]
        n = points.shape[0]

        cloud_arr = np.zeros(n, dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('r', np.uint8), ('g', np.uint8), ('b', np.uint8),
        ])
        cloud_arr['x'], cloud_arr['y'], cloud_arr['z'] = points[:, 0], points[:, 1], points[:, 2]
        cloud_arr['r'], cloud_arr['g'], cloud_arr['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = cloud_arr.dtype.itemsize
        msg.row_step = msg.point_step * n
        msg.is_dense = True
        msg.data = cloud_arr.tobytes()
        return msg


    def fusion_callback(self, lidar_msg: PointCloud2, image_msg: CompressedImage):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to decode RGB compressed image: {e}')
            return

        points_lidar = self.pointcloud2_to_xyz(lidar_msg)
        u, v, z = self.project_to_image(points_lidar, img.shape)
        sparse_depth, valid_mask = self.build_sparse_depth(u, v, z, img.shape)

        header = image_msg.header

        if self.do_sparse:
            sparse_msg = self.bridge.cv2_to_imgmsg(sparse_depth.astype(np.float32), encoding='32FC1')
            sparse_msg.header = header
            self.sparse_pub.publish(sparse_msg)

        dense_depth = None
        if self.do_dense or self.do_cloud:
            dense_depth = self.densify(sparse_depth, valid_mask, img)

            if self.depth_bias_m != 0.0:
                valid = np.isfinite(dense_depth) & (dense_depth > 0.0)
                dense_depth[valid] += self.depth_bias_m

            if self.do_dense:
                dense_msg = self.bridge.cv2_to_imgmsg(dense_depth.astype(np.float32), encoding='32FC1')
                dense_msg.header = header
                self.dense_pub.publish(dense_msg)

        if self.do_overlay:
            overlay = self.make_overlay(img, u, v, z)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = header
            self.overlay_pub.publish(overlay_msg)

        if self.do_cloud and dense_depth is not None:
            cloud_msg = self.backproject_colorized_cloud(dense_depth, img, header)
            self.cloud_pub.publish(cloud_msg)

        self._frame_count += 1
        if self.log_every_n and self._frame_count % self.log_every_n == 0:
            self.get_logger().info(
                f'frame {self._frame_count}: {points_lidar.shape[0]} lidar pts, '
                f'{u.size} projected in-bounds'
            )


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
