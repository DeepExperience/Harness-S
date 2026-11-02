"""Offline EC2 authentication endpoint for Moto's mock lifecycle tests.

Only the public transport outside Moto is replaced. Active Moto interceptors
still execute repository code, and no test outcome or assertion is rewritten.
"""
from urllib.parse import urlparse


def pytest_configure(config):
    from botocore.awsrequest import AWSResponse
    from botocore.httpsession import URLLib3Session
    original = URLLib3Session.send

    class Body:
        def stream(self, amt=None, decode_content=False):
            yield (b'<?xml version="1.0" encoding="UTF-8"?>'
                   b'<Response><Errors><Error><Code>AuthFailure</Code>'
                   b'<Message>AWS was not able to validate the provided access credentials</Message>'
                   b'</Error></Errors><RequestID>offline-authentication-fixture</RequestID></Response>')

    def send(self, request):
        host = urlparse(request.url).hostname
        if host == 'ec2.us-west-1.amazonaws.com':
            return AWSResponse(request.url, 401, {'content-type': 'text/xml'}, Body())
        return original(self, request)

    URLLib3Session.send = send
    config.add_cleanup(lambda: setattr(URLLib3Session, 'send', original))
