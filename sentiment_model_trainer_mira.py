from sentiment_model_trainer_base import SentimentModelTrainerBase

import numpy as np


class SentimentModelTrainerMIRA(SentimentModelTrainerBase):

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
                y_tag_fv = self.features_extractor.evaluate_document_feature_vector_summed(document, y_tag)

                G[next_G_line_idx, :] = (y_fv - y_tag_fv)
                next_G_line_idx += 1

                y_tag_loss = self.calc_labeling_loss(y, y_tag)
                L.append(y_tag_loss)
        L = np.array(L).reshape(-1, )
        return G, L
