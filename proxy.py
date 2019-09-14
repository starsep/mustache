import typing

import mitmproxy.addonmanager
import mitmproxy.connections
import mitmproxy.http
import mitmproxy.log
import mitmproxy.tcp
import mitmproxy.websocket
import mitmproxy.proxy.protocol

with open('dickbutt.jpg', 'rb') as f:
    d = f.read()

def response(flow: mitmproxy.http.HTTPFlow):
    headers = flow.response.headers
    content_type = headers["Content-Type"]
    print(f"Content type: {content_type}")
    if content_type in ("image/jpeg", "image/png"):
        print("Intercepting")
        flow.response = mitmproxy.http.HTTPResponse.make(
            200, d, {
                "Content-Type": "image/jpeg"
            }
        )
