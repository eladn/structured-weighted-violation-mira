from sentiment_model_trainer_base import SentimentModelTrainerBase
from document import Document

import numpy as np
import cvxpy as cvx
from warnings import warn
import sys


class SentimentModelTrainerSWVM(SentimentModelTrainerBase):

    def weights_update_step_on_batch(self, previous_w: np.ndarray, documents_batch: list,
                                     feature_vectors_batch: list, inferred_labelings_batch: list):
        """SWVM constraints set (the actual SWVM part) are not implemented yet."""

        new_w_variable = cvx.Variable(previous_w.shape)
        w_diff = new_w_variable - previous_w
        w_norm2 = cvx.pnorm(w_diff, p=2) ** 2
        objective = cvx.Minimize(w_norm2)

        constraints = []
        for document, feature_vector_summed, inferred_labelings in zip(
                documents_batch, feature_vectors_batch, inferred_labelings_batch):
            doc_constraints = self.calc_constraints_for_document(
                document, feature_vector_summed, inferred_labelings, new_w_variable)
            constraints += doc_constraints

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

    def calc_constraints_for_document(self, document: Document, y_fv: np.ndarray,
                                      inferred_labelings: list, new_w_variable: cvx.Variable):
        y = document.y()

        losses_list = []
        weighted_structured_fv_diff_list = []

        for y_tag in inferred_labelings:
            weighted_structured_fv_diff, gamma_denominator = self.calc_weighted_structured_fv_diff(
                document, y, y_tag, y_fv, new_w_variable)
            weighted_structured_fv_diff_list.append(
                cvx.reshape(weighted_structured_fv_diff, (1, weighted_structured_fv_diff.size))
            )
            losses_list.append(self.calc_labeling_loss(y, y_tag))  # gamma_denominator *

        L = np.array(losses_list)
        weighted_structured_fv_diff_matrix = cvx.vstack(weighted_structured_fv_diff_list)

        lhs_constraint_expr = cvx.matmul(weighted_structured_fv_diff_matrix, new_w_variable)
        constraints = [lhs_constraint_expr >= L]
        return constraints

    @staticmethod
    def iterate_over_mj(y: list, y_tag: list, copy: bool = False):
        m_j = list(y)
        for idx, y_tag_val in enumerate(y_tag):
            if y[idx] == y_tag_val:
                continue
            m_j[idx] = y_tag_val
            yield m_j

            if copy:
                m_j = list(y)
            else:
                m_j[idx] = y[idx]

    def calc_gamma_denominator(self, phi_mj_deltas: list, new_w_variable: cvx.Variable):
        gamma_denominator = cvx.Constant(0)
        for phi_mj_delta in phi_mj_deltas:
            dot = cvx.matmul(new_w_variable, phi_mj_delta)
            gamma_denominator += (-1) * cvx.min(dot, 0)
        return gamma_denominator

    def calc_weighted_structured_fv_diff(self, document: Document, y: list, y_tag: list, y_fv: np.ndarray,
                                         new_w_variable: cvx.Variable):
        phi_mj_deltas = [
            self.features_extractor.evaluate_document_feature_vector_summed(document, mj) - y_fv
            for mj in self.iterate_over_mj(y, y_tag)
        ]

        gamma_denominator = self.calc_gamma_denominator(phi_mj_deltas, new_w_variable)

        weighted_structured_fv_diff = cvx.Constant(0)
        for phi_mj_delta in phi_mj_deltas:
            gamma_mj_nominator = (-1) * cvx.min(cvx.matmul(new_w_variable, phi_mj_delta), 0)
            gamma_mj = gamma_mj_nominator  # / gamma_denominator  # TODO: consider multiply the loss L by this value (instead of dividing by it)
            weighted_structured_fv_diff += gamma_mj * phi_mj_delta

        return weighted_structured_fv_diff, gamma_denominator
