from optimizers import OptimizerBase
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from warnings import warn
import sys


class QPOptimizer(OptimizerBase):
    def find_next_weights_according_to_constraints(self, previous_w: np.ndarray,
                                                   G: np.ndarray, L: np.ndarray):
        """ G*w >= L """

        P = sparse.eye(previous_w.size)
        q = (np.copy(previous_w) * -1).reshape(-1, )
        h = L.reshape(-1, ) * -1
        G = G * -1
        G = csr_matrix(G)

        next_w = solve_qp(P, q, G, h, solver="osqp")
        if np.any(np.equal(next_w, None)):
            print(file=sys.stderr)
            warn_msg = "QP solver returned `None`s solution vector. Weights vector `w` has not been updated."
            warn(warn_msg)
            return None
        return next_w
