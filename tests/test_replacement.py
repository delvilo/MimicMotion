"""Unit tests for replacement utilities in mimicmotion.utils.replacement."""
import unittest
import numpy as np

from mimicmotion.utils.replacement import box_iou


class BoxIoUTests(unittest.TestCase):
    def test_identical_boxes(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([0, 0, 10, 10])
        self.assertAlmostEqual(box_iou(a, b), 1.0)

    def test_disjoint_boxes(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        self.assertAlmostEqual(box_iou(a, b), 0.0)

    def test_partial_overlap(self):
        # Box A area = 100 ([0,0] to [10,10])
        # Box B area = 100 ([5,0] to [15,10])
        # Overlap = [5,0] to [10,10], area = 50
        # Union = 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3 ~ 0.3333333...
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 0, 15, 10])
        self.assertAlmostEqual(box_iou(a, b), 1.0 / 3.0)

    def test_nested_boxes(self):
        # Outer box area = 100 ([0,0] to [10,10])
        # Inner box area = 16 ([3,3] to [7,7])
        # Intersection = 16
        # Union = 100
        # IoU = 16 / 100 = 0.16
        outer = np.array([0, 0, 10, 10])
        inner = np.array([3, 3, 7, 7])
        self.assertAlmostEqual(box_iou(outer, inner), 0.16)

    def test_touching_boxes_edge(self):
        # Adjacent boxes sharing an edge
        a = np.array([0, 0, 10, 10])
        b = np.array([10, 0, 20, 10])
        self.assertAlmostEqual(box_iou(a, b), 0.0)

    def test_touching_boxes_corner(self):
        # Adjacent boxes sharing a corner point
        a = np.array([0, 0, 10, 10])
        b = np.array([10, 10, 20, 20])
        self.assertAlmostEqual(box_iou(a, b), 0.0)

    def test_zero_area_boxes(self):
        # Point/line boxes with zero area
        a = np.array([0, 0, 0, 0])
        b = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(box_iou(a, b), 0.0)

    def test_float_coordinates(self):
        a = np.array([0.5, 0.5, 10.5, 10.5])
        b = np.array([5.5, 0.5, 15.5, 10.5])
        self.assertAlmostEqual(box_iou(a, b), 1.0 / 3.0)

    def test_return_type(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([0, 0, 10, 10])
        result = box_iou(a, b)
        self.assertIsInstance(result, float)


if __name__ == '__main__':
    unittest.main()
