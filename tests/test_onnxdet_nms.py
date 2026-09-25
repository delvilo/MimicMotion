import unittest
import numpy as np
from mimicmotion.dwpose.onnxdet import nms, multiclass_nms


class TestNMS(unittest.TestCase):
    def test_nms_single_box(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        nms_thr = 0.5
        keep = nms(boxes, scores, nms_thr)
        self.assertEqual(keep, [0])

    def test_nms_non_overlapping_boxes(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        ], dtype=np.float32)
        scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
        nms_thr = 0.5
        keep = nms(boxes, scores, nms_thr)
        # Should be ordered by scores descending: index 1 (0.9), then index 0 (0.8), then index 2 (0.7)
        self.assertEqual(keep, [1, 0, 2])

    def test_nms_overlapping_boxes_suppression(self):
        # Box 0 and Box 1 overlap heavily (IoU > 0.5)
        boxes = np.array([
            [0, 0, 10, 10],   # area = 121
            [1, 1, 10, 10],   # area = 100, intersection = 100, IoU = 100 / (121 + 100 - 100) = 100/121 = 0.826
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        nms_thr = 0.5
        keep = nms(boxes, scores, nms_thr)
        self.assertEqual(keep, [0])

    def test_nms_overlapping_boxes_below_threshold(self):
        # Box 0 and Box 1 overlap slightly (IoU < 0.5)
        boxes = np.array([
            [0, 0, 10, 10],   # area = 121 (x: 0..10, y: 0..10)
            [8, 8, 18, 18],   # area = 121 (x: 8..18, y: 8..18), intersection area = (10-8+1)*(10-8+1) = 3*3 = 9
        ], dtype=np.float32)  # IoU = 9 / (121 + 121 - 9) = 9 / 233 ~= 0.0386 < 0.5
        scores = np.array([0.9, 0.8], dtype=np.float32)
        nms_thr = 0.5
        keep = nms(boxes, scores, nms_thr)
        self.assertEqual(keep, [0, 1])

    def test_nms_identical_boxes(self):
        boxes = np.array([
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
        ], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        nms_thr = 0.5
        keep = nms(boxes, scores, nms_thr)
        # Highest score is index 1 (0.9), others are identical so IoU = 1.0 > 0.5, suppressed
        self.assertEqual(keep, [1])

    def test_nms_empty_input(self):
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        nms_thr = 0.5
        keep = nms(boxes, scores, nms_thr)
        self.assertEqual(keep, [])


class TestMulticlassNMS(unittest.TestCase):
    def test_multiclass_nms_standard(self):
        # 3 boxes, 2 classes
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 10, 10],
            [50, 50, 60, 60],
        ], dtype=np.float32)
        # scores shape: (N, num_classes) -> (3, 2)
        scores = np.array([
            [0.9, 0.1],  # Box 0: class 0 high score
            [0.8, 0.2],  # Box 1: class 0 high score (overlaps with box 0)
            [0.2, 0.95], # Box 2: class 1 high score
        ], dtype=np.float32)

        nms_thr = 0.5
        score_thr = 0.5

        dets = multiclass_nms(boxes, scores, nms_thr=nms_thr, score_thr=score_thr)
        # Output dets format: [x1, y1, x2, y2, score, cls_ind]
        self.assertIsNotNone(dets)
        self.assertEqual(dets.shape[0], 2)  # Box 1 suppressed for class 0, Box 0 kept for cls 0, Box 2 kept for cls 1

        # Class 0 detection: Box 0 (score 0.9)
        # Class 1 detection: Box 2 (score 0.95)
        cls_0_dets = dets[dets[:, 5] == 0]
        cls_1_dets = dets[dets[:, 5] == 1]

        self.assertEqual(len(cls_0_dets), 1)
        np.testing.assert_array_equal(cls_0_dets[0, :4], boxes[0])
        self.assertAlmostEqual(cls_0_dets[0, 4], 0.9)

        self.assertEqual(len(cls_1_dets), 1)
        np.testing.assert_array_equal(cls_1_dets[0, :4], boxes[2])
        self.assertAlmostEqual(cls_1_dets[0, 4], 0.95)

    def test_multiclass_nms_cross_class_overlap(self):
        # Two overlapping boxes, but strong predictions for different classes
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 10, 10],
        ], dtype=np.float32)
        scores = np.array([
            [0.9, 0.1],  # Box 0: class 0
            [0.1, 0.85], # Box 1: class 1
        ], dtype=np.float32)

        dets = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)
        self.assertIsNotNone(dets)
        self.assertEqual(len(dets), 2)  # Both kept since class-aware

    def test_multiclass_nms_no_valid_scores(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ], dtype=np.float32)
        scores = np.array([
            [0.1, 0.2],
            [0.3, 0.1],
        ], dtype=np.float32)

        dets = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)
        self.assertIsNone(dets)


if __name__ == "__main__":
    unittest.main()
