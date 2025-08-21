
'''
Abstraction for stream vs other services here
'''
from getstream import Stream


class EdgeTransport:
    pass


class StreamEdge(EdgeTransport):
    client: Stream

    def __init__(self, **kwargs):
        # Initialize Stream client
        self.client = Stream.from_env()
