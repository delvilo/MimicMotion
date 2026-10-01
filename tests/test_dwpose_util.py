import unittest
import numpy as np

from mimicmotion.dwpose.util import draw_bodypose, draw_handpose, draw_pose, LIMB_INDICES_0IDX, HAND_EDGE_COLORS


class TestDWPoseUtil(unittest.TestCase):
    def test_constants_precomputed(self):
        self.assertEqual(len(LIMB_INDICES_0IDX), 19)
        self.assertEqual(len(HAND_EDGE_COLORS), 20)
        self.assertTrue(all(isinstance(idx, np.ndarray) for idx in LIMB_INDICES_0IDX))

    def test_draw_bodypose_and_handpose(self):
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        candidate = np.array([[0.5, 0.5] for _ in range(18)])
        subset = np.array([[i for i in range(18)]])
        score = np.array([[0.9 for _ in range(18)]])

        res_body = draw_bodypose(canvas.copy(), candidate, subset, score)
        self.assertEqual(res_body.shape, (100, 100, 3))

        hand_peaks = [np.array([[0.5, 0.5] for _ in range(21)])]
        hand_scores = [np.array([0.9 for _ in range(21)])]
        res_hand = draw_handpose(canvas.copy(), hand_peaks, hand_scores)
        self.assertEqual(res_hand.shape, (100, 100, 3))

    def test_draw_pose(self):
        pose = {
            'bodies': {
                'candidate': np.array([[0.5, 0.5] for _ in range(18)]),
                'subset': np.array([[i for i in range(18)]]),
                'score': np.array([[0.9 for _ in range(18)]]),
            },
            'faces': [np.array([[0.5, 0.5] for _ in range(68)])],
            'faces_score': [np.array([0.9 for _ in range(68)])],
            'hands': [np.array([[0.5, 0.5] for _ in range(21)])],
            'hands_score': [np.array([0.9 for _ in range(21)])],
        }
        res = draw_pose(pose, H=100, W=100, ref_w=100)
        self.assertEqual(res.shape, (3, 100, 100))


if __name__ == "__main__":
    unittest.main()
