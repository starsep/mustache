#!/usr/bin/env python
import math
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import dlib
import numpy as np
from imutils import face_utils
from PIL import Image
from tqdm import tqdm

FACIAL_LANDMARKS_IDXS = OrderedDict(
    [("mouth", (48, 68)), ("inner_mouth", (60, 68)), ("nose", (27, 35))]
)


def loadAsRGBA(path: str):
    img = cv2.imread(path)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def loadStreamAsRGBA(stream):
    data = np.fromstring(stream, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def resizeMustache(mustache, size):
    mustache_resized = cv2.resize(mustache, size)
    return mustache_resized


def cvRGBAToPillow(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    im_pil = Image.fromarray(img2).convert("RGBA")
    return im_pil


def findCenterOfShape(label: str, shape):
    (start, end) = FACIAL_LANDMARKS_IDXS[label]
    points = shape[start:end]
    avgX = int(np.average([q[0] for q in points]))
    avgY = int(np.average([q[1] for q in points]))
    return (avgX, avgY)


def findMinMaxOfShape(label: str, shape):
    (start, end) = FACIAL_LANDMARKS_IDXS[label]
    points = shape[start:end]
    minX = np.min([q[0] for q in points])
    maxX = np.max([q[0] for q in points])
    minY = np.min([q[1] for q in points])
    maxY = np.max([q[1] for q in points])
    return (minX, maxX, minY, maxY)


def centerBetweenNoseAndMouth(img, shape):
    (noseAvgX, noseAvgY) = findCenterOfShape("nose", shape)
    (innerMouthAvgX, innerMouthAvgY) = findCenterOfShape("inner_mouth", shape)
    # debugMustacheCenter(img, noseAvgX, noseAvgY)
    # debugMustacheCenter(img, innerMouthAvgX, innerMouthAvgY)
    (vectorX, vectorY) = (innerMouthAvgX - noseAvgX, innerMouthAvgY - noseAvgY)
    angle = 0
    if vectorX != 0 and vectorY != 0:
        angle = 90 - np.rad2deg(math.atan2(vectorY, vectorX))
    noseBottom = findMinMaxOfShape("nose", shape)[3]
    (mouthLeft, mouthRight, _, mouthTop) = findMinMaxOfShape("mouth", shape)
    return (
        (noseAvgX + innerMouthAvgX) // 2,
        (noseAvgY + innerMouthAvgY) // 2,
        mouthTop - noseBottom,
        int(1.5 * (mouthRight - mouthLeft)),
        angle,
    )


def debugFaceRect(rect, img):
    rectP = (
        (rect.tl_corner().x, rect.tl_corner().y),
        (rect.br_corner().x, rect.br_corner().y),
    )
    cv2.rectangle(img, rectP[0], rectP[1], [0, 0, 255, 255])


def debugMustacheCenter(img, avgX, avgY):
    cv2.circle(img, (avgX, avgY), 1, [0, 0, 255, 255])


def rotateBound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def addMustaches(img, gray, rects, predictor, mustache, path: Path):
    imPil = cvRGBAToPillow(img)
    # print(f"Found: {len(rects)} faces")
    for (i, rect) in enumerate(rects):
        # debugFaceRect(rect, img)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (avgX, avgY, h, w, angle) = centerBetweenNoseAndMouth(img, shape)
        # debugMustacheCenter(img, avgX, avgY)
        mustacheResized = resizeMustache(mustache, (w, h))
        mustacheRotated = rotateBound(mustacheResized, -angle)
        mustachePil = cvRGBAToPillow(mustacheRotated)
        (left, top) = (
            avgX - mustacheRotated.shape[1] // 2,
            avgY - mustacheRotated.shape[0] // 2,
        )
        imPil.paste(mustachePil, (left, top), mustachePil)
        # cv2.imwrite(f'debug-{path.name}.jpg', img)
    if path is not None:
        imPil.convert("RGB").save(Path("output") / path.name)
    imgCv = np.array(imPil.convert("RGB"))
    return cv2.imencode(".jpg", imgCv)[1]


def mustachify(img, path: Path, mustache, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    result = addMustaches(img, gray, rects, predictor, mustache, path)
    return result.tostring()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
    for path in tqdm(sys.argv[1:]):
        path = Path(path)
        img = loadAsRGBA(str(path))
        mustachify(img, path, mustache, detector, predictor)
