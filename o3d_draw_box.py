# -*- coding:utf-8 -*- #
"""
-------------------------------------------------------------------
   Description :   draw 3d boxes use open3d
   File Name：     o3d_draw_box.py
   Author :        wdian
   create date：   2021/7/29
-------------------------------------------------------------------
"""

import os
import pickle
import colorsys
import open3d as o3d
import numpy as np


class DrawBoxes(object):
    """
    X(float32) Y(float32) Z(float32)
    w(float32) l(float32) h(float32)
    r(float32) label(float32) score(float32)
    """

    def __init__(self, point_file: str, class_name: list):
        self.class_name = class_name
        self._point_file = point_file
        self._pcd = o3d.geometry.PointCloud()
        self._vis = o3d.visualization.Visualizer()

        self.addPoint()

    def __del__(self):
        self._vis.destroy_window()

    def save_view_point(self, json_file):
        self._vis.get_render_option().point_size = 1
        self._vis.get_render_option().background_color = np.asarray([0, 0, 0])
        self._vis.run()  # user changes the view and press "q" to terminate
        param = self._vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(json_file, param)
        self._vis.clear_geometries()
        self._vis.destroy_window()

    @staticmethod
    def generate_colors(num_classes: int):
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]

        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(23)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors

    @staticmethod
    def processPont(point_file):
        data = np.fromfile(point_file, dtype=np.float32)
        data = data.reshape((-1, 4))
        data = data[:, :3].astype(np.float32)
        return data

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    def my_compute_box_3d(self, center, size, heading_angle):
        h = size[2]
        w = size[0]
        l = size[1]
        heading_angle = -heading_angle - np.pi / 2

        center[2] = center[2]  # + h / 2
        r = self.rotz(1 * heading_angle)
        l = l / 2
        w = w / 2
        h = h / 2
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(r, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

    def addPoint(self):
        data = self.processPont(self._point_file)
        self._vis.create_window(window_name="draw_boxes")

        self._pcd.points = o3d.utility.Vector3dVector(data)
        self._pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self._vis.add_geometry(self._pcd)

    def addBoxes(self, boxes, labels, color=None):
        if color is None:
            colors = self.generate_colors(len(self.class_name))
        else:
            colors = [color] * len(self.class_name)

        print("Detection boxes number: {}".format(len(boxes)))
        for i in range(len(boxes)):
            bbox = boxes[i]
            cls_index = int(labels[i])
            class_name = self.class_name[cls_index]

            corners_3d = self.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])

            bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                          [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

            cls_colors = [colors[cls_index] for _ in range(len(bbox_lines))]

            bbox = o3d.geometry.LineSet()
            bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
            bbox.colors = o3d.utility.Vector3dVector(cls_colors)

            bbox.points = o3d.utility.Vector3dVector(corners_3d)

            self._vis.add_geometry(bbox)

    def showBox(self):
        opt = self._vis.get_render_option()
        ctr = self._vis.get_view_control()

        opt.point_size = 1.2
        opt.background_color = np.asarray([0, 0, 0])
        # ctr.set_lookat([0, 0, 0.3])
        param = o3d.io.read_pinhole_camera_parameters('./data/ScreenView.json')
        self._vis.add_geometry(self._pcd)

        ctr.convert_from_pinhole_camera_parameters(param)
        self._vis.poll_events()
        self._vis.update_renderer()
        self._vis.run()


if __name__ == '__main__':
    file_index = '001250'
    dataset_path = "./data/point_cloud/"

    res_file1 = "./data/ground_truth-{}.bin".format(file_index)

    res_file2 = "./data/output-{}.bin".format(file_index)

    point_path = os.path.join(dataset_path, "{}.bin".format(file_index))
    class_names = ['car', 'bicycle', 'bus', 'motorcycle', 'pedestrian', 'truck']

    obj = DrawBoxes(point_path, class_name=class_names)
    # obj.save_view_point("./data/ScreenView.json")

    result_data_1 = np.fromfile(res_file1, np.float32)
    result_data_1 = np.array(result_data_1.reshape((-1, 9)))

    result_data_2 = np.fromfile(res_file2, np.float32)
    result_data_2 = np.array(result_data_2.reshape((-1, 12)))

    obj.addBoxes(boxes=result_data_1[:, 0:7],
                 labels=result_data_1[:, 7],
                 color=[1, 0, 0])

    obj.addBoxes(boxes=result_data_2[:, 0:7],
                 labels=result_data_2[:, 7],
                 color=[0, 1, 0])
    obj.showBox()
