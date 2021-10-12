import numpy as np
from ignite.metrics import Metric


class CodeMeter(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._codes = None
        super(CodeMeter, self).__init__(output_transform=output_transform)

    def reset(self):
        self._codes = []

    def update(self, output):
        self._codes += output

    def compute(self):
        return np.array(self._codes)
