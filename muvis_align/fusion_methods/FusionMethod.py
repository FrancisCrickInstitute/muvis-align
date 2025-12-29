from abc import ABC


class FusionMethod(ABC):
    def __init__(self, params, debug=False):
        self.params = params
        self.debug = debug
