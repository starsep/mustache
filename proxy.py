import typing

import mitmproxy.addonmanager
import mitmproxy.connections
import mitmproxy.http
import mitmproxy.log
import mitmproxy.tcp
import mitmproxy.websocket
import mitmproxy.proxy.protocol

import cv2
import dlib

from mustache import mustachify, loadStreamAsRGBA

mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def response(flow: mitmproxy.http.HTTPFlow):
    headers = flow.response.headers
    content_type = headers["Content-Type"]
    print(f"Content type: {content_type}")
    if content_type in ("image/jpeg", "image/png"):
        image = flow.response.data.content
        img = loadStreamAsRGBA(image)
        image_mustaches = mustachify(img, None, mustache, detector, predictor)
        print("Intercepting")
        flow.response = mitmproxy.http.HTTPResponse.make(
            200, image_mustaches, {
                "Content-Type": "image/jpeg"
            }
        )
