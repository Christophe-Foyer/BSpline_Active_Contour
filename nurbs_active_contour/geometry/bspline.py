"""
Author: Christophe Foyer

Description:
    This script extends and redefines the base NURBS-Python curve class to be 
    locked to degree 2 and forces a periodic constraint similar to that found
    in scipy's definition of a BSpline curve.
"""

from geomdl import BSpline
import numpy as np
from nurbs_active_contour.geometry.evaluators import CRQCurveEvaluator


class ClosedRationalQuadraticCurve(BSpline.Curve):

    _weights = None
    rational = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluator = CRQCurveEvaluator()
        self._degree[0] = 2

    @property
    def degree(self):
        return self._degree[0]

    @degree.setter
    def degree(self, value):
        print("Degree is locked to 2.")

    def set_ctrlpts(self, value):
        ctrlpts = np.array(value)
        degree = self.degree

        closed = all((ctrlpts[:degree] == ctrlpts[-degree:]).flatten())

        if not closed:
            ctrlpts = np.vstack([ctrlpts, ctrlpts[:degree]])

        if (not hasattr(self, '_weights')) or (self.weights is None):
            self.weights = [np.ones((len(ctrlpts))).tolist()]

        assert len(ctrlpts) == len(self.weights[0]), \
            "Control points and weights do not match"

        ctrlpts = np.hstack([ctrlpts,
                             np.array(self.weights[0]).reshape(-1, 1)])

        super().set_ctrlpts(ctrlpts.tolist())

    @property
    def weights(self):
        if self._weights:
            return self._weights[0]
        else:
            return None

    @weights.setter
    def weights(self, value):
        self._weights = [value]
