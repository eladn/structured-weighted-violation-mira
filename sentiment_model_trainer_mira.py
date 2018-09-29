from sentiment_model_trainer_base import SentimentModelTrainerBase

from scipy import sparse
import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from warnings import warn
import sys


class SentimentModelTrainerMIRA(SentimentModelTrainerBase):

    def weights_update_step_on_batch(self, w, documents_batch: list, feature_vectors_batch: list,
                                     inferred_labelings_batch: list):
        P, q, G, h = self.extract_qp_matrices(
            w, documents_batch, feature_vectors_batch, inferred_labelings_batch)
        next_w = solve_qp(P, q, G, h, solver="osqp")
        if np.any(np.equal(next_w, None)):
            print(file=sys.stderr)
            warn_msg = "QP solver returned `None`s solution vector. Weights vector `w` has not been updated."
            warn(warn_msg)
            return None
        return next_w

    def extract_qp_matrices(self, w, documents_batch: list, feature_vectors_batch: list,
                            inferred_labelings_batch: list):
        M = sparse.eye(self.features_extractor.nr_features)
        q = np.copy(w) * -1
        nr_labelings = sum(len(labelings) for labelings in inferred_labelings_batch)
        G = np.zeros((nr_labelings, self.features_extractor.nr_features))
        h = []

        next_G_line_idx = 0
        for document, feature_vector_summed, inferred_labelings in zip(documents_batch,
                                                                       feature_vectors_batch,
                                                                       inferred_labelings_batch):
            y = document.y()
            y_fv = feature_vector_summed
            for y_tag in inferred_labelings:
                y_tag_fv = self.features_extractor.evaluate_document_feature_vector_summed(document, y_tag)

                G[next_G_line_idx, :] = (y_tag_fv - y_fv)
                next_G_line_idx += 1

                y_tag_loss = self.calc_labeling_loss(y, y_tag)
                h.append(-y_tag_loss)
        G = csr_matrix(G)
        h = np.array(h).reshape(-1, )
        return M, q.reshape(self.features_extractor.nr_features, ), G, h
