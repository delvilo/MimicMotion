import math
import numpy as np
import matplotlib
import cv2


eps = 0.01

LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18], [3, 17], [6, 18]]
# Precompute 0-indexed limb indices for body pose drawing to avoid repeated array conversions inside loops
LIMB_INDICES_0IDX = [np.array(limb) - 1 for limb in LIMB_SEQ]

HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
              [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
# Precompute HSV-to-RGB color array for hand pose edges to avoid repeated color conversions in inner loops
HAND_EDGE_COLORS = [matplotlib.colors.hsv_to_rgb([ie / float(len(HAND_EDGES)), 1.0, 1.0]) for ie in range(len(HAND_EDGES))]


def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # Performance optimization: use precomputed 0-indexed limb indices
    for i in range(17):
        limb_idx = LIMB_INDICES_0IDX[i]
        for n in range(len(subset)):
            index = subset[n][limb_idx]
            conf = score[n][limb_idx]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            idx_int = index.astype(int)
            Y = candidate[idx_int, 0] * float(W)
            X = candidate[idx_int, 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    H, W, C = canvas.shape

    # Performance optimization: use precomputed edge colors and edges
    for peaks, scores in zip(all_hand_peaks, all_hand_scores):

        for ie, e in enumerate(HAND_EDGES):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), 
                         HAND_EDGE_COLORS[ie] * score, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores):
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'])

    ########################################### draw hand pose #####################################################
    canvas = draw_handpose(canvas, hands, pose['hands_score'])

    ########################################### draw face pose #####################################################
    canvas = draw_facepose(canvas, faces, pose['faces_score'])

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
