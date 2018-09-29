from sentiment_model_trainer_base import SentimentModelTrainerBase

import numpy as np
import cvxpy as cvx
from warnings import warn
import sys


class SentimentModelTrainerSWVM(SentimentModelTrainerBase):

    def weights_update_step_on_batch(self, previous_w: np.ndarray, documents_batch: list,
                                     feature_vectors_batch: list, inferred_labelings_batch: list):
        """SWVM constraints set (the actual SWVM part) are not implemented yet."""

        new_w = cvx.Variable(previous_w.shape)
        w_diff = new_w - previous_w
        w_norm2 = cvx.pnorm(w_diff, p=2) ** 2
        objective = cvx.Minimize(w_norm2)

        # TODO: implement the SWVM constraints set!
        constraints = []

        problem = cvx.Problem(objective, constraints)

        try:
            problem.solve()
        except cvx.SolverError as se:
            print(file=sys.stderr)
            warn_msg = "CVXPY solver raised `SolverError`. Weights vector `w` has not been updated. {}".format(se)
            warn(warn_msg)
            return None

        if problem.status in {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}:
            return new_w.value

        print(file=sys.stderr)
        warn_msg = "CVXPY solver couldn't solve the problem. " \
                   "Problem status is `{}`. Weights vector `w` has not been updated.".format(problem.status)
        warn(warn_msg)
        return None
