"""
Author: Christophe Foyer

Description:
    This script extends and redefines the base NURBS-Python evaluator classes
    to provide a higher performance function handle for faster curve evaluation
"""


from geomdl.evaluators import CurveEvaluator as geomdl_CurveEvaluator
from geomdl.evaluators import SurfaceEvaluatorRational
from geomdl import helpers
# import autograd.numpy as np
import numpy as np


class CRQCurveEvaluator(geomdl_CurveEvaluator):
    """
    This evaluator class offers a function handle that evaluates the Curve's 
    points faster than the pure-python implementation.
    """

    def evaluate(self, datadict, **kwargs):
        """ Evaluates the curve.
        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation
        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        # Geometry data from datadict
        degree = datadict['degree'][0]
        knotvector = np.array(datadict['knotvector'][0])
        ctrlpts = np.array(datadict['control_points'])
        size = datadict['size'][0]
        sample_size = datadict['sample_size'][0]
        dimension = datadict['dimension'] + 1 if datadict['rational'] \
            else datadict['dimension']

        # Split weights
        weights = ctrlpts[:, -1]
        ctrlpts = ctrlpts[:, :-1]

        # Keyword arguments
        start = kwargs.get('start', 0.0)
        stop = kwargs.get('stop', 1.0)

        # Algorithm A3.1
        knots = np.linspace(start, stop, sample_size)
        spans = helpers.find_spans(degree, knotvector, size, knots,
                                   self._span_func)
        basis = helpers.basis_functions(degree, knotvector, spans, knots)

        closed = kwargs.get('closed', False)
        if closed:
            assert all((ctrlpts[:degree] == ctrlpts[-degree:]).flatten()), \
                "Loop is not closed"
            ctrlpts = np.array(ctrlpts)[:-degree]

        def evaluate_points(ctrlpts):
            eval_points = np.zeros((len(knots), 2))
            for idx in range(len(knots)):
                crvpt = [0.0 for _ in range(dimension)]
                for i in range(0, degree + 1):
                    crvpt[:] = [crv_p + (basis[idx][i] * ctl_p)
                                for crv_p, ctl_p in
                                zip(crvpt, ctrlpts[spans[idx] - degree + i])]

                eval_points[idx, :] = crvpt
            return eval_points

        self.function = lambda ctrlpts: evaluate_points(ctrlpts)

        return self.function(ctrlpts).tolist()
    
    
class RQSurfEvaluator(SurfaceEvaluatorRational):
    """
    This class is unfinished, was meant to speed up evaluation as it does above
    """
    
    def evaluate(self, datadict, **kwargs):
        """ Evaluates the surface.
        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation
        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        # Geometry data from datadict
        sample_size = datadict['sample_size']
        degree = datadict['degree']
        knotvector = datadict['knotvector']
        ctrlpts = datadict['control_points']
        size = datadict['size']
        dimension = datadict['dimension'] + 1 if datadict['rational'] else datadict['dimension']
        pdimension = datadict['pdimension']
        precision = datadict['precision']

        # Keyword arguments
        start = kwargs.get('start', [0.0 for _ in range(pdimension)])
        stop = kwargs.get('stop', [1.0 for _ in range(pdimension)])

        # Algorithm A3.5
        spans = [[] for _ in range(pdimension)]
        basis = [[] for _ in range(pdimension)]
        for idx in range(pdimension):
            knots = np.linalg.linspace(start[idx], stop[idx], sample_size[idx], decimals=precision)
            spans[idx] = helpers.find_spans(degree[idx], knotvector[idx], size[idx], knots, self._span_func)
            basis[idx] = helpers.basis_functions(degree[idx], knotvector[idx], spans[idx], knots)

        eval_points = []
        for i in range(len(spans[0])):
            idx_u = spans[0][i] - degree[0]
            for j in range(len(spans[1])):
                idx_v = spans[1][j] - degree[1]
                spt = [0.0 for _ in range(dimension)]
                for k in range(0, degree[0] + 1):
                    temp = [0.0 for _ in range(dimension)]
                    for l in range(0, degree[1] + 1):
                        temp[:] = [tmp + (basis[1][j][l] * cp) for tmp, cp in
                                   zip(temp, ctrlpts[idx_v + l + (size[1] * (idx_u + k))])]
                    spt[:] = [pt + (basis[0][i][k] * tmp) for pt, tmp in zip(spt, temp)]

                eval_points.append(spt)

        return eval_points
