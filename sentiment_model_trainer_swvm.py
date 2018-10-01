from sentiment_model_trainer_base import SentimentModelTrainerBase
from document import Document

import numpy as np


class SentimentModelTrainerSWVM(SentimentModelTrainerBase):

    def calc_constraints_for_update_step_on_batch(self, previous_w: np.ndarray, documents_batch: list,
                                                  feature_vectors_batch: list, inferred_labelings_batch: list):
        nr_labelings = sum(len(labelings) for labelings in inferred_labelings_batch)
        G = np.zeros((nr_labelings, self.features_extractor.nr_features))
        L = []

        next_G_line_idx = 0
        for document, feature_vector_summed, inferred_labelings in zip(documents_batch,
                                                                       feature_vectors_batch,
                                                                       inferred_labelings_batch):
            y = document.y()
            y_fv = feature_vector_summed
            for y_tag in inferred_labelings:
                G_line = self.calc_weighted_structured_fv_diff(document, y, y_tag, y_fv, previous_w)
                G[next_G_line_idx, :] = G_line
                next_G_line_idx += 1

                y_tag_loss = self.calc_labeling_loss(y, y_tag)
                L.append(y_tag_loss)

        L = np.array(L).reshape(-1, )
        return G, L

    @staticmethod
    def iterate_over_mj(y: list, y_tag: list, copy: bool = False):
        m_j = list(y)
        labeling_nr = 0
        for idx, y_tag_val in enumerate(y_tag):
            if y[idx] == y_tag_val:
                continue
            m_j[idx] = y_tag_val
            labeling_nr += 1
            yield m_j

            if copy:
                m_j = list(y)
            else:
                m_j[idx] = y[idx]

        # yield also the actual `y_tag` labeling if needed
        if labeling_nr > 1:
            m_j = y_tag
            if copy:
                m_j = list(m_j)
            yield m_j

    def calc_gamma_denominator(self, phi_mj_deltas: list, previous_w: np.ndarray):
        gamma_denominator = 0
        for phi_mj_delta in phi_mj_deltas:
            dot_product = np.dot(previous_w, phi_mj_delta)
            gamma_denominator += (-1) * np.min(dot_product, 0)
        return gamma_denominator

    def calc_weighted_structured_fv_diff(self, document: Document, y: list, y_tag: list, y_fv: np.ndarray,
                                         previous_w: np.ndarray):
        phi_mj_deltas = [
            y_fv - self.features_extractor.evaluate_document_feature_vector_summed(document, mj)
            for mj in self.iterate_over_mj(y, y_tag)
        ]

        gamma_denominator = self.calc_gamma_denominator(phi_mj_deltas, previous_w)

        weighted_structured_fv_diff = 0
        for phi_mj_delta in phi_mj_deltas:
            if np.abs(gamma_denominator) < 0.00001:
                gamma_mj = 1 / len(phi_mj_deltas)
            else:
                gamma_mj_nominator = (-1) * np.min(np.dot(previous_w, phi_mj_delta), 0)
                gamma_mj = gamma_mj_nominator / gamma_denominator

            weighted_structured_fv_diff += gamma_mj * phi_mj_delta

        return weighted_structured_fv_diff
