from optimizers import OptimizerBase
import numpy as np
import cvxpy as cvx
from warnings import warn
import sys


class CVXOptimizer(OptimizerBase):
    def find_next_weights_according_to_constraints(self, previous_w: np.ndarray,
                                                   G: np.ndarray, L: np.ndarray):
        """ G*w >= L """

        new_w_variable = cvx.Variable(previous_w.shape)
        w_diff = new_w_variable - previous_w
        w_norm2 = cvx.pnorm(w_diff, p=2) ** 2
        objective = cvx.Minimize(w_norm2)

        lhs_constraint_expr = cvx.matmul(G, new_w_variable)
        constraints = [lhs_constraint_expr >= L]

        problem = cvx.Problem(objective, constraints)

        try:
            problem.solve()
        except cvx.SolverError as se:
            print(file=sys.stderr)
            warn_msg = "CVXPY solver raised `SolverError`. Weights vector `w` has not been updated. {}".format(se)
            warn(warn_msg)
            return None

        if problem.status in {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}:
            return new_w_variable.value

        print(file=sys.stderr)
        warn_msg = "CVXPY solver couldn't solve the problem. " \
                   "Problem status is `{}`. Weights vector `w` has not been updated.".format(problem.status)
        warn(warn_msg)
        return None
