from abc import ABC


class FusionMethod(ABC):
    def __init__(self, source, params, debug=False):
        self.source_type = source.dtype
        self.params = params
        self.debug = debug
