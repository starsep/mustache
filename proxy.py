import cv2
import dlib
import mitmproxy.addonmanager
import mitmproxy.connections
import mitmproxy.http
import mitmproxy.log
import mitmproxy.proxy.protocol
import mitmproxy.tcp
import mitmproxy.websocket
from mitmproxy.script import concurrent

from mustache import loadStreamAsRGBA, mustachify

mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


@concurrent
def response(flow: mitmproxy.http.HTTPFlow):
    headers = flow.response.headers
    content_type = headers["Content-Type"]
    print(f"Content type: {content_type}")
    if content_type in ("image/jpeg", "image/png"):
        image = flow.response.data.content
        img = loadStreamAsRGBA(image)
        image_mustaches = mustachify(img, None, mustache, detector, predictor)
        if image_mustaches is None:
            image_mustaches = image
        print("Intercepting")
        flow.response = mitmproxy.http.HTTPResponse.make(
            200, image_mustaches, {"Content-Type": "image/jpeg"}
        )
