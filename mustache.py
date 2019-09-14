#!/usr/bin/env python

import cv2
import numpy as np
from PIL import Image
import dlib
from collections import OrderedDict
from imutils import face_utils
import imutils
import sys
from pathlib import Path

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("nose", (27, 35)),
])

def loadAsRGBA(path: str):
    img = cv2.imread(path)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def findFaces(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

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

def centerBetweenNoseAndMouth(shape):
    (noseAvgX, noseAvgY) = findCenterOfShape("nose", shape)
    (innerMouthAvgX, innerMouthAvgY) = findCenterOfShape("inner_mouth", shape)
    noseBottom = findMinMaxOfShape("nose", shape)[3]
    (mouthLeft, mouthRight, _, mouthTop) = \
        findMinMaxOfShape("mouth", shape)
    return (
        (noseAvgX + innerMouthAvgX) // 2,
        (noseAvgY + innerMouthAvgY) // 2,
        mouthTop- noseBottom,
       int(1.5 * (mouthRight - mouthLeft))
    )

def debugFaceRect(rect, img):
    rectP = (
        (rect.tl_corner().x, rect.tl_corner().y),
        (rect.br_corner().x, rect.br_corner().y),
    )
    cv2.rectangle(img, rectP[0], rectP[1], [0, 0, 255, 255])

def debugMustacheCenter(img, avgX, avgY):
    cv2.circle(img, (avgX, avgY), 5, [0, 255, 0, 255])

def addMustaches(img, gray, rects, predictor, mustache, path: str):
    im_pil = cvRGBAToPillow(img)
    for (i, rect) in enumerate(rects):
        print(f'Found: {i}')
        # debugFaceRect(rect, img)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        clone = img.copy()
        (avgX, avgY, h, w) = centerBetweenNoseAndMouth(shape)
        # debugMustacheCenter(img, avgX, avgY)
        (left, top) = (avgX  - w // 2, avgY - h // 2)
        # cv2.rectangle(img, (left, top), (left + w, top + h), [0, 0, 255])
        mustache_resized = resizeMustache(mustache, (w, h))
        mustache_pil = cvRGBAToPillow(mustache_resized)
        im_pil.paste(mustache_pil, (left, top), mustache_pil)

    im_pil.convert("RGB").save(Path('output') / path.name)


def run(path: Path, mustache):
    img = loadAsRGBA(str(path))
    faces = findFaces(img)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    rects = detector(gray, 1)
    addMustaches(img, gray, rects, predictor, mustache, path)

if __name__ == '__main__':
    mustache = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)
    for path in sys.argv[1:]:
        run(Path(path), mustache)
